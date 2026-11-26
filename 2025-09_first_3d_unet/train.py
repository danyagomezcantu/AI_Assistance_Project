# src/train.py
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import VoxPairDataset
from model_3dcnn import MaskGenerator3DCNN

# ---- configs ----
ROOT = Path(__file__).resolve().parents[1]
VOX_DIR = ROOT / "data" / "processed" / "voxels"
CKPT_DIR = ROOT / "experiments" / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = ROOT / "experiments" / "logs"; LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH = 2
EPOCHS = 20
LR = 1e-3
TEST_FRACTION = 0.2

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def dice_iou(pred_bin: torch.Tensor, target: torch.Tensor):
    # both [B,1,D,H,W] in {0,1}
    inter = (pred_bin * target).sum(dim=(2,3,4))
    union = ((pred_bin + target) > 0).sum(dim=(2,3,4))
    dice = (2*inter) / (pred_bin.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-8)
    iou  = inter / (union + 1e-8)
    return dice.mean().item(), iou.mean().item()

def main():
    set_seed()
    ids = sorted({p.stem.split("_")[0] for p in VOX_DIR.glob("*_face.npy")})
    assert ids, f"No voxel files in {VOX_DIR}"
    # split by patient
    n_test = max(1, int(len(ids) * TEST_FRACTION))
    random.shuffle(ids)
    test_ids = ids[:n_test]; train_ids = ids[n_test:]
    print(f"Train {len(train_ids)} | Test {len(test_ids)}")

    train_ds = VoxPairDataset(VOX_DIR, train_ids)
    test_ds  = VoxPairDataset(VOX_DIR, test_ids)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH)

    device = torch.device("cpu")
    model = MaskGenerator3DCNN().to(device)
    crit = nn.BCEWithLogitsLoss()
    opt  = optim.Adam(model.parameters(), lr=LR)

    best_dice, best_path = -1.0, None
    for epoch in range(1, EPOCHS+1):
        model.train(); run_loss = 0.0
        for face, mask, _ in train_dl:
            face, mask = face.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(face)
            loss = crit(logits, mask)
            loss.backward(); opt.step()
            run_loss += loss.item() * face.size(0)
        train_loss = run_loss / len(train_ds)

        # eval
        model.eval(); run_loss = 0.0; dices=[]; ious=[]
        with torch.no_grad():
            for face, mask, _ in test_dl:
                face, mask = face.to(device), mask.to(device)
                logits = model(face)
                loss = crit(logits, mask); run_loss += loss.item() * face.size(0)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                d,i = dice_iou(pred, mask); dices.append(d); ious.append(i)
        test_loss = run_loss / len(test_ds)
        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_iou  = float(np.mean(ious)) if ious else 0.0

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | test {test_loss:.4f} | dice {mean_dice:.3f} | iou {mean_iou:.3f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_path = CKPT_DIR / f"best_dice_{best_dice:.3f}.pt"
            torch.save({"model": model.state_dict(),
                        "dice": best_dice}, best_path)
    print(f"Best dice {best_dice:.3f} saved to {best_path}")

if __name__ == "__main__":
    main()
