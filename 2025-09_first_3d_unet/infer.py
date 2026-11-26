# src/infer.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
from skimage import measure

from preprocess import load_mesh, pca_align, center, scale_to_unit, voxelize
from model_3dcnn import MaskGenerator3DCNN

ROOT = Path(__file__).resolve().parents[1]
VOX_DIR = ROOT / "data" / "processed" / "voxels"
TR_DIR  = ROOT / "data" / "processed" / "transforms"
PRED_DIR= ROOT / "data" / "processed" / "predictions"; PRED_DIR.mkdir(parents=True, exist_ok=True)

def predict_for_id(pid: str, ckpt_path: Path):
    # load face volume
    fpath = VOX_DIR / f"{pid}_face.npy"
    vol = np.load(fpath).astype("float32")[None,None]  # [1,1,D,H,W]
    face_t = torch.from_numpy(vol)

    # load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MaskGenerator3DCNN(); model.load_state_dict(ckpt["model"]); model.eval()

    with torch.no_grad():
        logits = model(face_t)
        probs = torch.sigmoid(logits)[0,0].numpy()
        pred = (probs >= 0.5).astype(np.uint8)

    # marching cubes
    verts, faces, _, _ = measure.marching_cubes(pred, level=0.5)
    mesh = trimesh.Trimesh(verts, faces, process=True)

    # invert transforms
    tr = json.loads((TR_DIR / f"{pid}.json").read_text())
    R = np.array(tr["R"]); pca_mean = np.array(tr["pca_mean"])
    center_vec = np.array(tr["center"]); s = float(tr["scale"])

    # undo scale & center
    mesh.vertices = mesh.vertices / s + center_vec
    # undo PCA rotation: reverse of (x - mean)@R^T + mean  ⇒  (x - mean)@R + mean
    mesh.vertices = (mesh.vertices - pca_mean) @ R + pca_mean

    out = PRED_DIR / f"{pid}_mask_pred.stl"
    mesh.export(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    # example:
    ckpt = list((ROOT / "experiments" / "checkpoints").glob("best_*.pt"))[-1]
    predict_for_id("00001", ckpt)
