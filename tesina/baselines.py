"""
baselines.py — P4: predictores de referencia no aprendidos B1 y B2 (brief §2.3).

Ambos se ajustan SOLO con los pacientes de entrenamiento de cada pliegue y
escriben sus predicciones en el mismo formato que la U-Net, para que pasen
por exactamente la misma ruta de evaluación.

B1, molde promedio: promedio vóxel a vóxel de los moldes de
    entrenamiento (en el marco alineado), umbralizado de modo que el número
    de vóxeles ocupados sea la mediana de los moldes de entrenamiento.
    El umbral fijo de 0.5 puede dejar el promedio vacío cuando los
    moldes varían mucho; igualar el volumen típico evita ese caso
    degenerado sin usar información del paciente de prueba.
B2, vecino más cercano: para cada paciente de prueba se busca el paciente
    de entrenamiento cuyo rostro se parece más (distancia euclidiana entre
    rostros suavizados y reducidos a 16³) y se devuelve su molde.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import Config
from .preprocess import load_arrays, usable_rows
from .train import pred_dir

log = logging.getLogger("tesina")


def _face_features(X: np.ndarray, target: int = 16) -> np.ndarray:
    G = X.shape[1]
    f = max(G // target, 1)
    feats = []
    for x in X:
        s = gaussian_filter(x.astype(np.float32), sigma=max(f / 2, 0.75))
        if f > 1:
            s = s.reshape(G // f, f, G // f, f, G // f, f).mean(axis=(1, 3, 5))
        feats.append(s.ravel())
    return np.stack(feats)


def mean_mask(Ytr: np.ndarray) -> np.ndarray:
    mean = Ytr.astype(np.float32).mean(axis=0)
    target = int(np.median(Ytr.reshape(len(Ytr), -1).sum(axis=1)))
    if target <= 0:
        return np.zeros_like(mean)
    thr = np.sort(mean.ravel())[::-1][min(target, mean.size) - 1]
    return (mean >= thr).astype(np.float32) * (mean > 0)


def run_baselines(cfg: Config, verbose: bool = True) -> dict:
    rows = usable_rows(cfg)
    tiempos = {}
    for fold in range(cfg.k_folds):
        tr = sorted(r["pid"] for r in rows if r["fold"] != fold)
        te = sorted(r["pid"] for r in rows if r["fold"] == fold)
        Xtr, Ytr = load_arrays(cfg, tr)
        Xte, _ = load_arrays(cfg, te)

        t0 = time.time()
        b1 = mean_mask(Ytr)
        d1 = pred_dir(cfg, "B1")
        d1.mkdir(parents=True, exist_ok=True)
        for pid in te:
            np.save(d1 / f"{pid}_prob.npy", b1.astype(np.float16))
        t_b1 = time.time() - t0

        t0 = time.time()
        Ftr, Fte = _face_features(Xtr), _face_features(Xte)
        d2 = pred_dir(cfg, "B2")
        d2.mkdir(parents=True, exist_ok=True)
        neighbours = {}
        for pid, f in zip(te, Fte):
            j = int(np.argmin(((Ftr - f) ** 2).sum(axis=1)))
            neighbours[pid] = tr[j]
            np.save(d2 / f"{pid}_prob.npy", Ytr[j].astype(np.float16))
        t_b2 = time.time() - t0
        tiempos[fold] = {"B1_s": round(t_b1, 3), "B2_s": round(t_b2, 3), "n_train": len(tr), "n_test": len(te),
                     "B1_vox": int(b1.sum())}
        (pred_dir(cfg, "B2") / f"vecinos_pliegue_{fold}.json").write_text(json.dumps(neighbours, indent=1))
        if verbose:
            log.info(f"[P4] pliegue {fold}: B1 ({int(b1.sum())} vóxeles) y B2 listos para {len(te)} pacientes")
    (pred_dir(cfg, "B1").parent / "baselines_tiempos.json").write_text(json.dumps(tiempos, indent=1))
    return tiempos
