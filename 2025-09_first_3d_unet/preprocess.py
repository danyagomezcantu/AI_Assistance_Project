# src/preprocess.py
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from skimage import measure

# ---------- math helpers ----------
def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b); s = np.linalg.norm(v)
    if s == 0:  # already aligned
        return np.eye(3, dtype=np.float64)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))

# ---------- core steps ----------
def load_mesh(stl_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Not a triangle mesh: {stl_path}")
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh

def pca_align(points: np.ndarray, target_axis=np.array([0,1,0])) -> tuple[np.ndarray, dict]:
    pca = PCA(n_components=3).fit(points)
    R = _rotation_matrix_from_vectors(pca.components_[0], target_axis)
    aligned = (points - pca.mean_) @ R.T + pca.mean_
    return aligned, {"R": R.tolist(), "pca_mean": pca.mean_.tolist()}

def center(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbmin, bbmax = points.min(axis=0), points.max(axis=0)
    c = (bbmax + bbmin) / 2.0
    return points - c, c

def scale_to_unit(points: np.ndarray) -> tuple[np.ndarray, float]:
    bb = points.max(axis=0) - points.min(axis=0)
    longest = float(np.max(bb))
    if longest == 0:
        return points.copy(), 1.0
    s = 1.0 / longest
    return points * s, s

def voxelize(mesh: trimesh.Trimesh, grid: int = 64, fill=True) -> tuple[np.ndarray, float]:
    # mesh is already normalized to ~unit box; pick pitch so unit ~ grid
    pitch = 1.0 / grid
    vg = mesh.voxelized(pitch=pitch)
    if fill:
        vg = vg.fill()
    return vg.matrix.astype(np.uint8), pitch

def marching_cubes_from_volume(vol: np.ndarray, level=0.5) -> trimesh.Trimesh:
    verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=level)
    return trimesh.Trimesh(verts, faces, process=True)

# ---------- pipeline for a face-mask pair ----------
def preprocess_pair(face_stl: Path, mask_stl: Path, out_dir: Path, grid: int = 64, save_mesh_preview=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vox_dir = out_dir / "voxels"; vox_dir.mkdir(exist_ok=True, parents=True)
    tr_dir = out_dir / "transforms"; tr_dir.mkdir(exist_ok=True, parents=True)
    mesh_dir = out_dir / "meshes"; mesh_dir.mkdir(exist_ok=True, parents=True)

    pid = face_stl.stem.split("_")[0]  # expects {ID}_face_scan.stl
    assert pid in mask_stl.stem, "Face/Mask IDs must match"

    face_mesh = load_mesh(face_stl)
    mask_mesh = load_mesh(mask_stl)

    # 1) PCA align using FACE, apply same R to MASK
    f_aligned, info = pca_align(face_mesh.vertices)
    R = np.array(info["R"], dtype=np.float64)
    m_aligned = (mask_mesh.vertices - np.array(info["pca_mean"])) @ R.T + np.array(info["pca_mean"])

    # 2) Center using combined bbox center
    combined = np.vstack([f_aligned, m_aligned])
    combined_centered, center_vec = center(combined)
    f_centered = f_aligned - center_vec
    m_centered = m_aligned - center_vec

    # 3) Scale both using FACE longest dim
    f_scaled, scale = scale_to_unit(f_centered)
    m_scaled = m_centered * scale

    # 4) Put back into meshes & voxelize
    face_mesh.vertices = f_scaled
    mask_mesh.vertices = m_scaled
    f_vol, pitch = voxelize(face_mesh, grid=grid, fill=True)
    m_vol, _     = voxelize(mask_mesh, grid=grid, fill=True)

    # save volumes (.npy)
    np.save(vox_dir / f"{pid}_face.npy", f_vol)
    np.save(vox_dir / f"{pid}_mask.npy", m_vol)

    # save transforms to invert later (JSON)
    tr = {
        "R": info["R"],
        "pca_mean": info["pca_mean"],
        "center": center_vec.tolist(),
        "scale": scale,
        "pitch": pitch,
        "grid": grid
    }
    with open(tr_dir / f"{pid}.json", "w") as f:
        json.dump(tr, f, indent=2)

    # optional: preview aligned meshes (STL) for sanity checks
    if save_mesh_preview:
        face_mesh.export(mesh_dir / f"{pid}_face_aligned.stl")
        mask_mesh.export(mesh_dir / f"{pid}_mask_aligned.stl")

def preprocess_all(raw_faces: Path, raw_masks: Path, out_dir: Path, grid: int = 64, save_mesh_preview=False):
    raw_faces = Path(raw_faces); raw_masks = Path(raw_masks)
    ids = []
    for face_stl in sorted(raw_faces.glob("*_face_scan.stl")):
        pid = face_stl.stem.split("_")[0]
        mask_stl = raw_masks / f"{pid}_mask.stl"
        if mask_stl.exists():
            preprocess_pair(face_stl, mask_stl, out_dir, grid=grid, save_mesh_preview=save_mesh_preview)
            ids.append(pid)
    print(f"Preprocessed {len(ids)} pairs.")
    return ids
