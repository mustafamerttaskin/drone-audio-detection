"""
cnn_model.py
------------
Kompakt bir CNN. Log-mel spektrogramı "tek kanallı imge" olarak alır ve
İHA / İHA-değil ikili sınıflandırması yapar.

Neden büyük hazır bir model (ResNet50 vs.) kullanmıyoruz?
--------------------------------------------------------
- Veri seti küçük (birkaç bin örnek); büyük model overfit eder.
- Model dizüstünde CPU'da bile makul sürede eğitilmeli.
- Aselsan/Baykar için "edge cihazda çalışan hafif model" hikayesi güçlü.

Mimari (yaklaşık 250-300 bin parametre):
  Conv-BN-ReLU-Pool  ×4
  Global average pooling
  Dropout → Linear (n_classes)

Global average pooling, giriş zaman uzunluğuna esneklik sağlar; sabit
uzunluk zorunluluğu ortadan kalkar.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class DroneCNN(nn.Module):
    """
    Küçük ama sağlam CNN. Girdi: (B, 1, n_mels, T). Çıktı: (B, n_classes).
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),    # n_mels/2, T/2
            ConvBlock(32, 64, pool=(2, 2)),   # /4
            ConvBlock(64, 128, pool=(2, 2)),  # /8
            ConvBlock(128, 128, pool=(2, 2)), # /16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
