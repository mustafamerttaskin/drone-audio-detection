"""
scripts/generate_synthetic_data.py
----------------------------------
Küçük bir SENTETİK veri seti üretir. Amaç, gerçek veri henüz
indirilmemişken pipeline'ın uçtan uca çalıştığını doğrulamaktır.

- "drone" örnekleri: 90-250 Hz temel frekans + harmonikler + hafif jitter
  (döner pervane akustik imzasının kaba yaklaşımı)
- "not_drone" örnekleri: bantlı beyaz gürültü + tonlu olmayan iz
  (arka plan / trafik / rüzgâr gibi)

NOT: Gerçek proje sonuçları için Sara Al-Emadi veri seti kullanılmalıdır.
Bu script yalnızca "smoke test" içindir.

Kullanım:
    python scripts/generate_synthetic_data.py --n 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
SR = 22050
DUR = 2.0


def synth_drone(rng: np.random.Generator) -> np.ndarray:
    """Sentetik drone benzeri ses: temel + harmonikler + jitter."""
    n = int(SR * DUR)
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 250)                    # temel frekans
    jitter = rng.uniform(0.5, 2.0)               # frekans dalgalanması
    f_t = f0 + jitter * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f_t) / SR
    y = np.zeros_like(t)
    # 5 harmonik, azalan genlik
    for k in range(1, 6):
        y += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    y *= 0.4
    y += 0.02 * rng.standard_normal(n)           # hafif motor gürültüsü
    return y.astype(np.float32)


def synth_not_drone(rng: np.random.Generator) -> np.ndarray:
    """Sentetik arka plan: bantlı gürültü + kısa tıklamalar."""
    n = int(SR * DUR)
    y = rng.standard_normal(n).astype(np.float32) * 0.15
    # birkaç kısa transient
    for _ in range(rng.integers(0, 4)):
        pos = int(rng.integers(0, n - 500))
        y[pos : pos + 500] += 0.2 * rng.standard_normal(500).astype(np.float32)
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30,
                        help="Her sınıf için üretilecek örnek sayısı")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for cls in ("drone", "not_drone"):
        (OUT / cls).mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        y_d = synth_drone(rng)
        sf.write(str(OUT / "drone" / f"synthetic_drone_{i:03d}.wav"), y_d, SR)

        y_n = synth_not_drone(rng)
        sf.write(str(OUT / "not_drone" / f"synthetic_not_drone_{i:03d}.wav"), y_n, SR)

    print(f"[OK] {args.n} + {args.n} sentetik örnek yazıldı: {OUT}")
    print("[UYARI] Bu veriler yalnızca pipeline testi içindir. "
          "Gerçek sonuçlar için Sara Al-Emadi veri seti kullanın.")


if __name__ == "__main__":
    main()
