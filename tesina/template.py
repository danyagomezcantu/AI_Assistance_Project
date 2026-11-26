"""
template.py — P8: derivación determinista de una plantilla básica (brief §1, §2.2[A]).

Actualización del 24 nov 2026: el archivo `*_mask.stl` resultó ser un molde
POSITIVO sólido (rostro modificado + base plana), no una mascarilla hueca. Por
eso la pieza S es directamente el sólido predicho, sin cavidad que rellenar.
Qué es exactamente `*_template.stl` en el archivo de AirTec sigue pendiente
de confirmar; mientras tanto, esta función deriva el NEGATIVO del molde:
  1. Pieza S = sólido predicho (molde positivo).
  2. Bloque B = caja alineada que contiene a S, ampliada `template_margin_mm`
     por lado (y recortada a la cuadrícula).
  3. Plantilla T = B ∖ S, exportada a STL en mm del paciente.
  4. Revisión de contrasalidas a lo largo del eje de desmolde (profundidad
     del rostro): una columna en la que S aparece en más de un tramo no se
     puede desmoldar en esa dirección con un molde de dos partes. Se reporta
     la fracción de columnas con contrasalida.
Ángulos de salida, base de vaciado y canales quedan como trabajo futuro.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from . import geometry as geo


def derive_template(mask_vox: np.ndarray, face_vox: np.ndarray, tf: geo.Transform,
                    margin_mm: float = 10.0, demold_axis: int = 2):
    S = mask_vox.astype(bool)
    if not S.any():
        return None, {"vacio": 1}
    G = S.shape[0]
    m = int(np.ceil(margin_mm / tf.pitch_mm))
    idx = np.argwhere(S)
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + m + 1, G)
    B = np.zeros_like(S)
    B[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    T = B & ~S
    lab, n_comp = label(T)

    # contrasalidas: número de tramos de S por columna a lo largo del eje de desmolde
    Sm = np.moveaxis(S, demold_axis, -1).astype(np.int8)
    starts = (np.diff(np.concatenate([np.zeros(Sm.shape[:-1] + (1,), np.int8), Sm], axis=-1), axis=-1) == 1).sum(-1)
    cols = starts > 0
    undercut = float((starts[cols] > 1).mean()) if cols.any() else 0.0

    mesh = geo.volume_to_mesh_mm(T.astype(np.float32), tf, 0.5, "taubin", 5)
    info = {
        "vacio": 0, "vox_molde": int(S.sum()),
        "volumen_molde_ml": round(float(S.sum() * tf.pitch_mm ** 3 / 1000), 1),
        "componentes_plantilla": int(n_comp), "frac_columnas_contrasalida": round(undercut, 4),
        "plantilla_watertight": int(mesh.is_watertight) if mesh is not None else 0,
    }
    return mesh, info
