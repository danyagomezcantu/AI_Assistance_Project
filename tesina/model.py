"""
model.py — U-Net 3D de la configuración C0 (Çiçek et al., 2016).

Entrada: ocupación del rostro (1 canal, G³). Salida: logits de ocupación de
la mascarilla (1 canal, G³). Cada nivel del codificador aplica dos
convoluciones 3×3×3 con normalización por instancia y LeakyReLU; el
decodificador sube con convolución transpuesta y concatena el mapa del mismo
nivel del codificador (conexiones de salto).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.down.append(ConvBlock(c, ch, dropout))
            c = ch
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose3d(c, ch, 2, stride=2))
            self.dec.append(ConvBlock(2 * ch, ch, dropout))
            c = ch
        self.head = nn.Conv3d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
