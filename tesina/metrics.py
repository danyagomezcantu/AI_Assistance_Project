"""
metrics.py — métricas volumétricas y de superficie en milímetros (brief §2.5).

Volumétricas (en la cuadrícula G³): Dice e IoU entre la ocupación predicha y
la ocupación de referencia preprocesada.

De superficie (en mm del paciente): se compara la malla predicha, ya llevada
de vuelta a milímetros, contra el STL ORIGINAL de la mascarilla de AirTec, de
modo que el error incluye también la pérdida por discretizar en vóxeles.

Regiones del molde de referencia (desde el 24 nov 2026)
-------------------------------------------------------
El archivo de AirTec es un molde positivo sólido con base plana. Cada punto
muestreado en la superficie de referencia recibe una región según su normal
respecto a la normal hacia afuera de la base (register.base_plane):
  * frente : normal opuesta a la base (n · n_base < −0.3); es la superficie
             que reproduce el rostro modificado por AirTec, la que importa;
  * base   : el resto (la base plana y las paredes laterales), que son
             artefactos de fabricación.
Los puntos de la malla predicha heredan la región del punto de referencia más
cercano. Para cada región R:
  d_ref→pred = distancia de cada punto de referencia en R a la predicción;
  d_pred→ref = distancia de cada punto predicho con región R a la referencia;
  Chamfer simétrico = (media(d_ref→pred) + media(d_pred→ref)) / 2;
  HD95 = máximo de los percentiles 95 de ambas direcciones.
Las distancias punto–superficie se aproximan con el vecino más cercano en un
muestreo denso (5 veces más puntos) de la superficie destino.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import sample_surface

REGIONS = ("total", "frente", "base")


def dice_iou(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool)
    r = ref.astype(bool)
    inter = np.logical_and(p, r).sum()
    s = p.sum() + r.sum()
    union = np.logical_or(p, r).sum()
    dice = 2.0 * inter / s if s else 1.0
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def label_regions(ref_normals: np.ndarray, base_normal: np.ndarray) -> np.ndarray:
    """1 = frente, 2 = base (incluye paredes laterales)."""
    return np.where(ref_normals @ base_normal < -0.3, 1, 2)


def surface_metrics(pred_mm: trimesh.Trimesh | None, ref_mm: trimesh.Trimesh,
                    n: int = 20000, seed: int = 0, cache: dict | None = None) -> dict:
    """Chamfer y HD95 por región. `cache` permite reusar el muestreo de la referencia."""
    out = {}
    if cache is None or "ref_pts" not in (cache or {}):
        ref_q, fi = sample_surface(ref_mm, n, seed)
        ref_dense, _ = sample_surface(ref_mm, 5 * n, seed + 1)
        from .register import base_plane
        labels = label_regions(ref_mm.face_normals[fi], base_plane(ref_mm)["normal"])
        ref_dense_tree = cKDTree(ref_dense)
        ref_q_tree = cKDTree(ref_q)
        c = {"ref_pts": ref_q, "labels": labels, "ref_dense_tree": ref_dense_tree, "ref_q_tree": ref_q_tree}
        if cache is not None:
            cache.update(c)
    else:
        c = cache
    ref_q, labels = c["ref_pts"], c["labels"]
    for k, name in enumerate(REGIONS):
        out[f"n_ref_{name}"] = int(len(labels) if k == 0 else (labels == k).sum())
    if pred_mm is None or len(pred_mm.faces) == 0:
        for name in REGIONS:
            out[f"chamfer_{name}"] = np.nan
            out[f"hd95_{name}"] = np.nan
        return out

    pred_q, _ = sample_surface(pred_mm, n, seed + 3)
    pred_dense, _ = sample_surface(pred_mm, 5 * n, seed + 4)
    d_ref2pred, _ = cKDTree(pred_dense).query(ref_q)
    d_pred2ref, _ = c["ref_dense_tree"].query(pred_q)
    _, nn = c["ref_q_tree"].query(pred_q)
    pred_lab = labels[nn]
    for k, name in enumerate(REGIONS):
        a = d_ref2pred if k == 0 else d_ref2pred[labels == k]
        b = d_pred2ref if k == 0 else d_pred2ref[pred_lab == k]
        if len(a) == 0 or len(b) == 0:
            out[f"chamfer_{name}"] = np.nan
            out[f"hd95_{name}"] = np.nan
            continue
        out[f"chamfer_{name}"] = float((a.mean() + b.mean()) / 2.0)
        out[f"hd95_{name}"] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
    return out
