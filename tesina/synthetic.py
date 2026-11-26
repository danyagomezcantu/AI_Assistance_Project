"""
synthetic.py — archivo sintético con la misma estructura que el de AirTec.

Sirve para probar el flujo completo sin un solo dato de paciente. Desde el
24 nov 2026 imita lo que se vio en el paciente real 01543:
  * rostro: superficie abierta tipo mapa de alturas con nariz, frente y
    mentón de tamaño aleatorio (STL abierto, como un escaneo). La nariz baja
    despacio hacia la frente y cae rápido hacia el labio, como en un rostro
    real, para que la regla de orientación tenga a qué agarrarse;
  * "mask": MOLDE POSITIVO sólido y cerrado. Su frente es el rostro en la
    región nariz–boca–mentón con modificaciones (nariz engrosada, zona de la
    boca hundida) y su parte trasera es una base plana;
  * el molde se guarda en OTRO sistema de coordenadas que el rostro (rotación
    y traslación aleatorias grandes), igual que en el archivo real, para que
    el registro rígido tenga que trabajar. `shared_frame=True` lo evita;
  * una plantilla de relleno (un bloque) y una ficha PNG vacía, solo para
    que el indexador encuentre los cuatro archivos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _face_height(x, y, p):
    """Altura z(x, y) del rostro sintético (mm)."""
    base = -(x / p["ax"]) ** 2 * 25 - (y / p["ay"]) ** 2 * 12
    dy = y - p["nose_y"]
    L = np.where(dy > 0, p["nose_l"] * 1.8, p["nose_l"] * 0.6)
    nose = p["nose_h"] * np.exp(-((x / p["nose_w"]) ** 2 + (dy / L) ** 2))
    brow = 6.0 * np.exp(-(((y - 40) / 10) ** 2)) * np.exp(-((x / 35) ** 2))
    chin = 4.0 * np.exp(-(((y + 45) / 8) ** 2)) * np.exp(-((x / 15) ** 2))
    return base + nose + brow + chin


def _modification(x, y, p):
    """Cambios de AirTec sobre el rostro: nariz engrosada (+) y boca hundida (−)."""
    nose = p["mod_nose"] * np.exp(-((x / (p["nose_w"] * 1.6)) ** 2 + ((y - p["nose_y"] + 4) / 16) ** 2))
    mouth = -p["mod_mouth"] * np.exp(-((x / 18) ** 2 + ((y + 28) / 7) ** 2))
    return nose + mouth


def make_patient(seed: int, shared_frame: bool = False):
    """Devuelve (rostro, molde, molde_en_marco_del_rostro) en mm."""
    rng = np.random.default_rng(seed)
    p = {
        "ax": rng.uniform(55, 70), "ay": rng.uniform(70, 90),
        "nose_h": rng.uniform(16, 24), "nose_w": rng.uniform(8, 12),
        "nose_l": rng.uniform(14, 20), "nose_y": rng.uniform(0, 8),
        "mod_nose": rng.uniform(3, 7), "mod_mouth": rng.uniform(1, 3),
    }
    # ---- rostro: mapa de alturas abierto (x en [-50,50], y en [-60,55])
    xs = np.arange(-50, 50.1, 2.0)
    ys = np.arange(-60, 55.1, 2.0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = _face_height(X, Y, p)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    nx, ny = X.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    face = trimesh.Trimesh(V, F, process=True)

    # ---- molde positivo: frente = rostro + modificaciones, base plana detrás
    rx, ry = rng.uniform(34, 40), rng.uniform(38, 44)
    cy = rng.uniform(-12, -8)
    h = 1.0
    gx = np.arange(-rx - 3, rx + 3.1, h)
    gy = np.arange(cy - ry - 3, cy + ry + 3.1, h)
    zf = _face_height(gx[:, None], gy[None, :], p)
    z_base = zf.min() - rng.uniform(8, 14)
    gz = np.arange(z_base - 3, zf.max() + 12, h)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    r2 = (GX / rx) ** 2 + ((GY - cy) / ry) ** 2
    front = _face_height(GX, GY, p) + _modification(GX, GY, p)
    f = np.minimum(np.minimum(front - GZ, GZ - z_base), (1 - r2) * 15.0)
    from skimage import measure
    fpad = np.pad(f, 1, constant_values=-1.0)
    verts, faces, _, _ = measure.marching_cubes(fpad, level=0.0, spacing=(h, h, h))
    verts = verts - h + np.array([gx[0], gy[0], gz[0]])
    mould = trimesh.Trimesh(verts, faces, process=True)
    mould.fix_normals()

    # ---- transformación del escáner del rostro
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rng.uniform(-25, 25, 3), degrees=True).as_matrix()
    T[:3, 3] = rng.uniform(-30, 30, 3) + np.array([0, 0, 900])
    face.apply_transform(T)
    mould.apply_transform(T)
    mould_true = mould.copy()
    if not shared_frame:   # el molde vive en otro marco, como en el archivo real
        T2 = np.eye(4)
        T2[:3, :3] = Rotation.random(random_state=seed).as_matrix()
        T2[:3, 3] = rng.uniform(-700, 700, 3)
        mould.apply_transform(T2)
    return face, mould, mould_true


def build_archive(root: str | Path, n_pairs: int = 20, n_only_face: int = 3, seed: int = 7) -> Path:
    """Escribe un archivo sintético con la taxonomía de carpetas de AirTec."""
    root = Path(root)
    pair_dir = root / "1. Training" / "complete_file_set"
    face_only = root / "3. Outliers" / "only_face"
    jaw = root / "4. Unterkiefer Files" / "99999"
    for d in (pair_dir, face_only, jaw):
        d.mkdir(parents=True, exist_ok=True)
    (jaw / "99999_mask.stl").write_bytes(b"solid x\nendsolid x\n")  # debe ignorarse
    png = b"\x89PNG\r\n\x1a\n"
    block = trimesh.creation.box(extents=(10, 10, 10))
    for i in range(n_pairs):
        pid = f"{90000 + i:05d}"
        face, mask, _ = make_patient(seed * 1000 + i)
        d = pair_dir / pid
        d.mkdir(exist_ok=True)
        face.export(d / f"{pid}_face_scan.stl")
        mask.export(d / f"{pid}_mask.stl")
        block.export(d / f"{pid}_template.stl")
        (d / f"{pid}_datasheet.png").write_bytes(png)
    for i in range(n_only_face):
        pid = f"{80000 + i:05d}"
        face, _, _ = make_patient(seed * 1000 + 500 + i)
        d = face_only / pid
        d.mkdir(exist_ok=True)
        face.export(d / f"{pid}_face_scan.stl")
    return root


if __name__ == "__main__":
    import sys
    out = build_archive(sys.argv[1] if len(sys.argv) > 1 else "/content/sintetico", 20)
    print("archivo sintético en", out)
