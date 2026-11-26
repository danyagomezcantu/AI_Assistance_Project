"""
train.py — P5/P6: entrenamiento de C0 por pliegue, reanudable, con registro de costo.

Para el pliegue f:
  * prueba        = pacientes con fold == f (nunca se tocan al entrenar);
  * validación    = 10 % de los demás, elegidos por SHA-256 (determinista),
                    solo para escoger la mejor época;
  * entrenamiento = el resto.
Cada época se guarda `last.pt` en Drive (modelo, optimizador, escalador,
historial, tiempo acumulado), y `best.pt` cuando mejora el Dice de
validación. Si Colab se desconecta, volver a llamar `train_fold` continúa
desde la última época guardada.

Pérdida: Dice suave + BCE con logits; el peso positivo de BCE es
sqrt(negativos/positivos) del conjunto de entrenamiento, para compensar que
la pared de la mascarilla ocupa una fracción pequeña de la cuadrícula sin
sobrecorregir.

Al terminar, predice el pliegue de prueba con `best.pt` y guarda las
probabilidades (float16) en predicciones/G{G}/{nombre}/{pid}_prob.npy, junto
con un resumen JSON con tiempo de pared y memoria pico de GPU (brief §3.2).
"""

from __future__ import annotations

import logging

import hashlib
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .model import UNet3D, count_parameters
from .preprocess import load_arrays, usable_rows

log = logging.getLogger("tesina")


def pred_dir(cfg: Config, method: str) -> Path:
    return cfg.work / "predicciones" / f"G{cfg.grid}" / method


def split_fold(cfg: Config, fold: int) -> tuple[list[str], list[str], list[str]]:
    rows = usable_rows(cfg)
    test = sorted(r["pid"] for r in rows if r["fold"] == fold)
    rest = sorted((r["pid"] for r in rows if r["fold"] != fold),
                  key=lambda p: hashlib.sha256(f"val:{cfg.fold_seed}:{p}".encode()).hexdigest())
    n_val = max(1, math.ceil(cfg.inner_val_fraction * len(rest))) if cfg.inner_val_fraction > 0 else 0
    return sorted(rest[n_val:]), sorted(rest[:n_val]), test


def dice_bce_loss(logits, target, pos_weight, dice_w=1.0, bce_w=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    dice = (2 * inter + 1.0) / (p.sum(dims) + target.sum(dims) + 1.0)
    return bce_w * bce + dice_w * (1 - dice.mean())


@torch.no_grad()
def predict(model, X: np.ndarray, device, amp: bool, batch: int = 2) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pr = torch.sigmoid(model(xb).float())
        out.append(pr.squeeze(1).cpu().numpy())
    return np.concatenate(out)


def _dice_np(p, y):
    p = p > 0.5
    y = y > 0.5
    s = p.sum() + y.sum()
    return 2 * np.logical_and(p, y).sum() / s if s else 1.0


# Campos que no se pueden cambiar a media corrida: mezclarían pliegues
# entrenados de forma distinta (y el calendario coseno guarda T_max).
_FIJOS = ("epochs", "base_channels", "depth", "lr", "weight_decay", "batch_size", "patience", "grid")


def _misma_configuracion(cfg: Config) -> None:
    previa = cfg.run_dir / "config.json"
    hay_avance = any(cfg.run_dir.glob("pliegue_*/last.pt")) or any(cfg.run_dir.glob("pliegue_*/resumen.json"))
    if not (previa.exists() and hay_avance):
        return
    vieja = json.loads(previa.read_text())
    nueva = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
    distintos = [f"{k}: antes {vieja[k]}, ahora {nueva[k]}" for k in _FIJOS
                 if k in vieja and k in nueva and vieja[k] != nueva[k]]
    if distintos:
        raise ValueError(
            f"{cfg.run_dir} ya tiene pliegues entrenados con otra configuración ("
            + "; ".join(distintos) + "). No se mezclan: regresa la primera celda a como estaba, "
            "o borra esa carpeta en Drive para empezar P6 de cero.")


def train_fold(cfg: Config, fold: int, device: str | None = None, verbose: bool = True) -> dict:
    cfg.validate()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = cfg.amp and dev.type == "cuda"
    _misma_configuracion(cfg)
    fdir = cfg.run_dir / f"pliegue_{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.json")
    summary_path = fdir / "resumen.json"
    if summary_path.exists():
        if verbose:
            log.info(f"[P6] pliegue {fold} ya terminado: {summary_path}")
        return json.loads(summary_path.read_text())

    tr, va, te = split_fold(cfg, fold)
    Xtr, Ytr = load_arrays(cfg, tr)
    Xva, Yva = load_arrays(cfg, va) if va else (None, None)
    pos = Ytr.mean()
    pw = cfg.bce_pos_weight or float(np.sqrt((1 - pos) / max(pos, 1e-6)))
    pos_weight = torch.tensor([pw], device=dev)

    torch.manual_seed(cfg.train_seed + fold)
    rng = np.random.default_rng(cfg.train_seed + fold)
    model = UNet3D(1, 1, cfg.base_channels, cfg.depth, cfg.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = {"epoch": 0, "best_val": -1.0, "best_epoch": -1, "history": [], "elapsed_s": 0.0,
             "peak_mem_mb": 0.0, "stale": 0}
    last = fdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        rng = np.random.default_rng(cfg.train_seed + fold + 1000 * state["epoch"])
        if verbose:
            log.info(f"[P6] pliegue {fold}: reanudando desde la época {state['epoch']}")

    if verbose:
        log.info(f"[P6] pliegue {fold}: {len(tr)} entrenamiento, {len(va)} validación, {len(te)} prueba; "
              f"{count_parameters(model):,} parámetros; dispositivo {dev}; pos_weight={pw:.2f}")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    Xtr_t = torch.from_numpy(Xtr).float().unsqueeze(1)
    Ytr_t = torch.from_numpy(Ytr).float().unsqueeze(1)
    while state["epoch"] < cfg.epochs:
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xtr_t[idx].to(dev, non_blocking=True), Ytr_t[idx].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            loss = dice_bce_loss(logits.float(), yb, pos_weight, cfg.dice_weight, cfg.bce_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        sched.step()

        val = float("nan")
        if Xva is not None:
            P = predict(model, Xva, dev, use_amp)
            val = float(np.mean([_dice_np(p, y) for p, y in zip(P, Yva)]))
        state["epoch"] += 1
        state["elapsed_s"] += time.time() - t0
        if dev.type == "cuda":
            state["peak_mem_mb"] = max(state["peak_mem_mb"], torch.cuda.max_memory_allocated(dev) / 2 ** 20)
        improved = (Xva is None) or (val > state["best_val"])
        if improved:
            state["best_val"], state["best_epoch"], state["stale"] = val, state["epoch"], 0
            torch.save(model.state_dict(), fdir / "best.pt")
        else:
            state["stale"] += 1
        state["history"].append({"epoch": state["epoch"], "loss": float(np.mean(losses)), "val_dice": val,
                                 "epoch_s": round(time.time() - t0, 2)})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, last)
        if verbose:
            log.info(f"[P6] pliegue {fold} época {state['epoch']}/{cfg.epochs}: pérdida {np.mean(losses):.4f}, "
                  f"Dice val {val:.4f}{' *' if improved else ''}, {time.time() - t0:.1f} s")
        if cfg.patience and state["stale"] >= cfg.patience:
            if verbose:
                log.info(f"[P6] parada temprana: {cfg.patience} épocas sin mejora")
            break

    # ---- predicción del pliegue de prueba con la mejor época
    model.load_state_dict(torch.load(fdir / "best.pt", map_location=dev, weights_only=True))
    Xte, _ = load_arrays(cfg, te)
    t0 = time.time()
    P = predict(model, Xte, dev, use_amp)
    infer_s = (time.time() - t0) / max(len(te), 1)
    pdir = pred_dir(cfg, cfg.name)
    pdir.mkdir(parents=True, exist_ok=True)
    for pid, p in zip(te, P):
        np.save(pdir / f"{pid}_prob.npy", p.astype(np.float16))

    summary = {
        "config": cfg.name, "grid": cfg.grid, "fold": fold, "n_train": len(tr), "n_val": len(va),
        "n_test": len(te), "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
        "best_val_dice": state["best_val"], "train_time_s": round(state["elapsed_s"], 1),
        "peak_gpu_mem_mb": round(state["peak_mem_mb"], 1), "infer_s_per_patient": round(infer_s, 4),
        "n_parameters": count_parameters(model), "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "pos_weight": pw,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (fdir / "historial.json").write_text(json.dumps(state["history"], indent=1), encoding="utf-8")
    if verbose:
        log.info(f"[P6] pliegue {fold} terminado: mejor época {state['best_epoch']}, "
              f"{summary['train_time_s'] / 60:.1f} min, memoria pico {summary['peak_gpu_mem_mb']:.0f} MB")
    return summary


def train_all(cfg: Config, folds=None, device: str | None = None) -> list[dict]:
    folds = range(cfg.k_folds) if folds is None else folds
    return [train_fold(cfg, f, device) for f in folds]


def pending_folds(cfg: Config) -> list[int]:
    return [f for f in range(cfg.k_folds) if not (cfg.run_dir / f"pliegue_{f}" / "resumen.json").exists()]
