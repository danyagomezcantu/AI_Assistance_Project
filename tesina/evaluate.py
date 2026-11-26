"""
evaluate.py — P4/P7: una sola ruta de evaluación para C0, B1 y B2.

Para cada método y cada paciente con predicción:
    probabilidades G³ → umbral → Dice/IoU contra la ocupación de referencia
    → marching cubes → suavizado (Taubin) → transformación inversa a mm
    → Chamfer y HD95 por región contra el STL original de AirTec.
Escribe una fila por (configuración, método, pliegue, paciente) en
resultados/metricas_por_paciente_G{G}.csv, con la configuración registrada.
Es reanudable: no recalcula filas que ya existen.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np

from . import geometry as geo
from .config import Config
from .metrics import dice_iou, surface_metrics
from .preprocess import _rss_gb, load_pair, paths_for, usable_rows
from .train import pred_dir

log = logging.getLogger("tesina")


def results_path(cfg: Config) -> Path:
    return cfg.results_dir / f"metricas_por_paciente_G{cfg.grid}.csv"


def _existing(path: Path) -> tuple[list[dict], set]:
    if not path.exists():
        return [], set()
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, {(r["method"], r["pid"]) for r in rows}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def smoothing_path(cfg: Config) -> Path:
    return cfg.results_dir / f"metricas_suavizado_G{cfg.grid}.csv"


def evaluate_methods(cfg: Config, methods: list[str], verbose: bool = True, save_every: int = 10,
                     smoothing: str | None = None, sufijo: str = "", out_path: Path | None = None) -> Path:
    """Evalúa las predicciones guardadas de `methods`.

    `smoothing`, `sufijo` y `out_path` sirven para evaluar las MISMAS
    predicciones con otro suavizado sin tocar la tabla principal: la fila se
    guarda con método = método + sufijo (por ejemplo «C0_sin_suavizado»).
    """
    smoothing = smoothing or cfg.smoothing
    out_path = out_path or results_path(cfg)
    rows, done = _existing(out_path)
    manifest = {r["pid"]: r for r in usable_rows(cfg)}
    cfg.save(cfg.results_dir / f"config_evaluacion_G{cfg.grid}.json")
    todo = [(m, pid) for pid in sorted(manifest) for m in methods
            if (m + sufijo, pid) not in done and (pred_dir(cfg, m) / f"{pid}_prob.npy").exists()]
    if verbose:
        log.info(f"[eval] {len(todo)} evaluaciones pendientes ({', '.join(m + sufijo for m in methods)}, "
                 f"suavizado {smoothing})")
    # agrupar por paciente para reusar el muestreo de la referencia
    by_pid: dict[str, list[str]] = {}
    for m, pid in todo:
        by_pid.setdefault(pid, []).append(m)
    n_new = 0
    n_err = 0
    for i, (pid, ms) in enumerate(by_pid.items(), 1):
        row = manifest[pid]
        try:
            # el molde de referencia se usa YA registrado al marco del rostro
            _, ref_mm, _ = load_pair(cfg, row)
            ref_mm.fix_normals()
            tf = geo.Transform.from_json(paths_for(cfg, pid)["tf"])
            ref_vox = np.load(paths_for(cfg, pid)["mask"])
        except Exception:
            log.exception(f"[eval] {pid}: no se pudo cargar la referencia; se salta")
            n_err += 1
            continue
        cache: dict = {}
        for m in ms:
            t0 = time.time()
            try:
                prob = np.load(pred_dir(cfg, m) / f"{pid}_prob.npy").astype(np.float32)
                prob = geo.clean_volume(prob, cfg.threshold, cfg.largest_component)
                binv = prob >= cfg.threshold
                dice, iou = dice_iou(binv, ref_vox)
                mesh = geo.volume_to_mesh_mm(prob, tf, cfg.threshold, smoothing, cfg.smooth_iterations)
                sm = surface_metrics(mesh, ref_mm, cfg.eval_samples, cfg.eval_seed, cache)
            except Exception:
                log.exception(f"[eval] {pid} con {m}: falló la evaluación; se salta")
                n_err += 1
                continue
            if mesh is None:
                log.warning(f"[eval] {pid} con {m + sufijo}: predicción vacía (métricas de superficie = NaN)")
            rows.append({
                "config": cfg.name if m not in ("B1", "B2") else m, "method": m + sufijo, "grid": cfg.grid,
                "fold": row["fold"], "pid": pid, "dice": round(dice, 5), "iou": round(iou, 5),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sm.items()},
                "empty_prediction": int(mesh is None), "pitch_mm": round(tf.pitch_mm, 4),
                "postproc_s": round(time.time() - t0, 3), "smoothing": smoothing,
            })
            n_new += 1
        # Las mallas de trimesh forman ciclos de referencias: sin esto, la memoria
        # crece ~0.35 GB por paciente hasta tumbar la sesión (P7, 25 nov 2026).
        ref_mm = mesh = cache = prob = binv = ref_vox = None
        gc.collect()
        if n_new and (i % save_every == 0):
            _write(out_path, rows)
        if verbose and (i % 10 == 0 or i == len(by_pid)):
            ram = _rss_gb(os.getpid())
            log.info(f"[eval] {i}/{len(by_pid)} pacientes"
                     + (f" (memoria en uso {ram:.1f} GB)" if ram else ""))
    _write(out_path, rows)
    if n_err:
        log.warning(f"[eval] {n_err} evaluaciones fallaron; las trazas están en la bitácora")
    if verbose:
        log.info(f"[eval] resultados en {out_path}")
    return out_path
