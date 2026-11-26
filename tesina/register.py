"""
register.py — llevar el molde al sistema de coordenadas del rostro (supuesto A1).

Qué se encontró (paciente 01543, 24 nov 2026)
---------------------------------------------
* El archivo `*_mask.stl` no es una mascarilla hueca: es un MOLDE POSITIVO
  sólido y cerrado, cuyo frente reproduce la región nariz–boca–mentón del
  rostro con modificaciones de AirTec (nariz engrosada, boca abierta,
  ranuras) y cuya parte trasera es una base plana.
* El rostro y el molde NO comparten sistema de coordenadas: el molde estaba
  a ~600 mm del rostro y rotado ~112°. Sin registro, todo par de
  entrenamiento estaría desalineado.

Qué hace este módulo
--------------------
1. `largest_component`: se queda con el cuerpo principal del molde (el STL
   trae astillas de 2 triángulos que no forman parte del sólido).
2. `base_plane`: encuentra la base plana del molde (la región plana de mayor
   área) y su normal hacia afuera. El frente del molde son las caras cuya
   normal apunta en sentido contrario a la base.
3. `register_mould_to_face`: registro rígido (rotación + traslación, sin
   escala) del frente del molde contra la superficie del rostro:
     a. inicialización: alinear la dirección del frente del molde con la del
        rostro (geometry.face_frame) y las puntas de nariz de ambos;
     b. barrido de 12 giros de 30° alrededor de ese eje, con ICP recortado
        corto en cada uno (el giro de 180° es ambiguo a simple vista);
     c. refinamiento de los 3 mejores con ICP punto–plano recortado
        (se usa solo el 60 % de pares más cercanos, porque el frente del
        molde difiere del rostro donde AirTec lo modificó);
     d. estabilidad: se repite el refinamiento con 40 % y 80 % de pares y se
        mide cuánto se mueven los puntos del molde (mm). Ese número es una
        cota práctica de la incertidumbre del registro.
   Si antes de registrar el molde ya está pegado al rostro (≥ 30 % de su
   frente a menos de 3 mm), se asume marco común y no se registra.

Limitación que hay que declarar en el documento: como el frente del molde no
es una copia exacta del rostro, el registro solo puede ser tan bueno como
las zonas no modificadas lo permitan; la incertidumbre por paciente queda
registrada y se reporta.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import face_frame, sample_surface

log = logging.getLogger("tesina")


def largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    main = max(parts, key=lambda c: len(c.faces))
    return main, len(parts) - 1


def base_plane(mesh: trimesh.Trimesh) -> dict:
    """Región plana de mayor área: normal hacia afuera, desplazamiento, área y planitud."""
    n = mesh.face_normals
    a = mesh.area_faces
    q = np.round(n * 20).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    best = int(np.argmax(np.bincount(inv, weights=a)))
    sel = inv == best
    nb = (n[sel] * a[sel, None]).sum(0)
    nb /= np.linalg.norm(nb)
    d = mesh.triangles_center[sel] @ nb
    return {"normal": nb, "offset": float(np.median(d)), "area_mm2": float(a[sel].sum()),
            "planitud_mm": float(d.std()), "frac_area": float(a[sel].sum() / a.sum())}


def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.clip(a @ b, -1, 1))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return Rotation.from_rotvec(np.pi * perp / np.linalg.norm(perp)).as_matrix()
    return Rotation.from_rotvec(v / s * np.arccos(c)).as_matrix()


def _icp(src, tree, tgt, tgt_n, R, t, iters, keep, plane):
    for _ in range(iters):
        X = src @ R.T + t
        dist, j = tree.query(X)
        s = dist <= np.quantile(dist, keep)
        p, q = X[s], tgt[j[s]]
        if plane:
            nn = tgt_n[j[s]]
            A = np.hstack([np.cross(p, nn), nn])
            b = np.einsum("ij,ij->i", q - p, nn)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            dR = Rotation.from_rotvec(x[:3]).as_matrix()
            dt = x[3:]
        else:
            ca, cb = p.mean(0), q.mean(0)
            U, _, Vt = np.linalg.svd((p - ca).T @ (q - cb))
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            dR = Vt.T @ D @ U.T
            dt = cb - dR @ ca
        R = dR @ R
        t = dR @ t + dt
    dist, _ = tree.query(src @ R.T + t)
    return R, t, dist


def register_mould_to_face(face: trimesh.Trimesh, mould: trimesh.Trimesh, mode: str = "auto",
                           seed: int = 0) -> tuple[np.ndarray, dict]:
    """Devuelve la matriz 4×4 que lleva el molde al marco del rostro y un diagnóstico."""
    rng = np.random.default_rng(seed)
    fp, fi = sample_surface(face, 80000, seed)
    fn = face.face_normals[fi]
    tree = cKDTree(fp)
    bp = base_plane(mould)
    nb = bp["normal"]
    mp, mi = sample_surface(mould, 80000, seed + 1)
    front = (mould.face_normals[mi] @ nb) < -0.3
    F = mp[front]
    info = {"base_area_mm2": round(bp["area_mm2"], 1), "base_planitud_mm": round(bp["planitud_mm"], 3),
            "frac_frente": round(float(front.mean()), 3)}

    d_raw, _ = tree.query(F[rng.choice(len(F), min(5000, len(F)), replace=False)])
    info["frac_frente_a_3mm_sin_registro"] = round(float((d_raw < 3).mean()), 4)
    if mode == "none" or (mode == "auto" and info["frac_frente_a_3mm_sin_registro"] >= 0.3):
        info["registrado"] = 0
        return np.eye(4), info

    R_face, _, finfo = face_frame(face, fp)
    d = R_face[2]
    nose_f = np.asarray(finfo["nariz_mm"])
    h = -(F @ nb)
    nose_m = F[h > h.max() - 3.0].mean(0)
    R0 = _rot_a_to_b(-nb, d)

    coarse_src = F[rng.choice(len(F), min(1200, len(F)), replace=False)]
    cands = []
    for ang in range(0, 360, 30):
        R = Rotation.from_rotvec(np.deg2rad(ang) * d).as_matrix() @ R0
        t = nose_f - R @ nose_m
        R, t, dist = _icp(coarse_src, tree, fp, fn, R, t, 12, 0.6, plane=False)
        cands.append((float(np.quantile(dist, 0.6)), ang, R, t))
    cands.sort(key=lambda c: c[0])

    src = F[rng.choice(len(F), min(4000, len(F)), replace=False)]
    best = None
    for _, ang, R, t in cands[:2]:
        R, t, dist = _icp(src, tree, fp, fn, R, t, 30, 0.6, plane=True)
        score = float(np.quantile(dist, 0.6))
        if best is None or score < best[0]:
            best = (score, ang, R, t, dist)
    score, ang, R, t, dist = best

    # estabilidad: cuánto se mueve el frente del molde al cambiar el recorte
    moves = []
    for keep in (0.4, 0.8):
        R2, t2, _ = _icp(src, tree, fp, fn, R.copy(), t.copy(), 20, keep, plane=True)
        moves.append(float(np.linalg.norm((src @ R2.T + t2) - (src @ R.T + t), axis=1).mean()))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    info.update({
        "registrado": 1, "giro_inicial_grados": ang,
        "residuo_mediana_mm": round(float(np.median(dist)), 3),
        "frac_frente_a_1mm": round(float((dist < 1).mean()), 4),
        "frac_frente_a_2mm": round(float((dist < 2).mean()), 4),
        "incertidumbre_mm": round(max(moves), 3),
        "rotacion_grados": round(float(np.degrees(Rotation.from_matrix(R).magnitude())), 2),
        "traslacion_mm": round(float(np.linalg.norm(t)), 1),
    })
    return T, info
