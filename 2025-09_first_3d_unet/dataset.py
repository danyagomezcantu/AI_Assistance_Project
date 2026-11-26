# src/dataset.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class VoxPairDataset(Dataset):
    def __init__(self, vox_dir: Path, id_list: list[str]):
        self.vox_dir = Path(vox_dir)
        self.ids = id_list

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        face = np.load(self.vox_dir / f"{pid}_face.npy")  # (Z,Y,X) boolean/uint8
        mask = np.load(self.vox_dir / f"{pid}_mask.npy")
        # to torch [1,D,H,W], float32
        face_t = torch.from_numpy(face[None].astype('float32'))
        mask_t = torch.from_numpy(mask[None].astype('float32'))
        return face_t, mask_t, pid
