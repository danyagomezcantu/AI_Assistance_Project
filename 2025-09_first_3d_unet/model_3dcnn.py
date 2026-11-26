# src/model_3dcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.c1 = nn.Conv3d(1, 8, 3, padding=1)
        self.c2 = nn.Conv3d(8, 16, 3, padding=1)
        self.p1 = nn.MaxPool3d(2)      # 64->32
        self.c3 = nn.Conv3d(16, 32, 3, padding=1)
        self.c4 = nn.Conv3d(32, 32, 3, padding=1)
        self.p2 = nn.MaxPool3d(2)      # 32->16
        # decoder
        self.u1 = nn.ConvTranspose3d(32, 32, 2, stride=2)   # 16->32
        self.dc1 = nn.Conv3d(64, 16, 3, padding=1)          # concat skip
        self.dc2 = nn.Conv3d(16, 16, 3, padding=1)
        self.u2 = nn.ConvTranspose3d(16, 16, 2, stride=2)   # 32->64
        self.dc3 = nn.Conv3d(32, 8, 3, padding=1)           # concat skip
        self.out = nn.Conv3d(8, 1, 3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.c1(x)); x2 = F.relu(self.c2(x1))
        y = self.p1(x2)
        x3 = F.relu(self.c3(y)); x4 = F.relu(self.c4(x3))
        y = self.p2(x4)
        y = F.relu(self.u1(y))
        y = torch.cat([y, x4], dim=1)
        y = F.relu(self.dc1(y)); y = F.relu(self.dc2(y))
        y = F.relu(self.u2(y))
        y = torch.cat([y, x2], dim=1)
        y = F.relu(self.dc3(y))
        return self.out(y)  # logits
