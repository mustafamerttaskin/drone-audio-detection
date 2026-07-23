"""
dataset.py
----------
PyTorch Dataset sınıfı. Klasör yapısı ImageFolder mantığındadır:

    data/raw/
      ├── drone/         *.wav
      └── not_drone/     *.wav

- Split (train/val/test) sınıf-katmanlı olarak (stratified) yapılır.
- Augmentasyon yalnızca eğitim setinde açıktır (`training=True`).
- CNN yolu için log-mel spektrogram, klasik ML yolu için feature vektörü
  aynı Dataset'ten farklı `feature_type` ile alınabilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.audio_utils import (
    add_gaussian_noise,
    load_audio,
    pitch_shift,
    time_shift,
)
from src.feature_extraction import (
    extract_feature_vector,
    extract_mel_spectrogram,
    normalize_spectrogram,
)


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class Sample:
    path: Path
    label: int  # 0 = not_drone, 1 = drone


def scan_dataset(data_root: str | Path, classes: List[str]) -> List[Sample]:
    """
    Sınıf klasörlerini gezip Sample listesi üretir.
    Sınıflar konfigürasyondaki sıraya göre etiketlenir.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Veri kökü bulunamadı: {data_root}")

    samples: List[Sample] = []
    for label_idx, class_name in enumerate(classes):
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Sınıf klasörü yok: {class_dir}. "
                f"Beklenen yapı: {data_root}/{class_name}/*.wav"
            )
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                samples.append(Sample(path=p, label=label_idx))

    if not samples:
        raise RuntimeError(f"{data_root} altında hiç ses dosyası bulunamadı.")
    return samples


def stratified_split(
    samples: List[Sample],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Sınıf oranlarını koruyarak train / val / test'e böler.
    """
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        samples, test_size=test_size, stratify=labels, random_state=random_state
    )
    # val boyutu, kalan setin oranına dönüştürülür
    val_relative = val_size / (1.0 - test_size)
    train_labels = [s.label for s in train_val]
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_labels,
        random_state=random_state,
    )
    return train, val, test


class DroneAudioDataset(Dataset):
    """
    Ses dosyalarını okuyup CNN veya klasik ML için öznitelik üreten
    tembel-yüklemeli Dataset.
    """

    def __init__(
        self,
        samples: List[Sample],
        audio_cfg: dict,
        feature_type: str = "mel",   # "mel" | "vector"
        training: bool = False,
        aug_cfg: dict | None = None,
        seed: int = 42,
    ):
        assert feature_type in {"mel", "vector"}, feature_type
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.feature_type = feature_type
        self.training = training
        self.aug_cfg = aug_cfg or {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------ #
    def _augment(self, y: np.ndarray) -> np.ndarray:
        if not (self.training and self.aug_cfg.get("enabled", False)):
            return y

        if self.rng.random() < self.aug_cfg.get("time_shift_prob", 0.0):
            y = time_shift(y, rng=self.rng)

        if self.rng.random() < self.aug_cfg.get("noise_prob", 0.0):
            lo, hi = self.aug_cfg.get("noise_snr_db", [10, 25])
            snr = float(self.rng.uniform(lo, hi))
            y = add_gaussian_noise(y, snr_db=snr, rng=self.rng)

        if self.rng.random() < self.aug_cfg.get("pitch_shift_prob", 0.0):
            lo, hi = self.aug_cfg.get("pitch_shift_semitones", [-2, 2])
            steps = float(self.rng.uniform(lo, hi))
            y = pitch_shift(y, sample_rate=self.audio_cfg["sample_rate"], n_steps=steps)

        return y

    # ------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        y = load_audio(
            path=sample.path,
            sample_rate=self.audio_cfg["sample_rate"],
            duration_sec=self.audio_cfg["duration_sec"],
            random_crop=self.training,
            rng=self.rng,
        )
        y = self._augment(y)

        if self.feature_type == "mel":
            spec = extract_mel_spectrogram(
                y=y,
                sample_rate=self.audio_cfg["sample_rate"],
                n_fft=self.audio_cfg["n_fft"],
                hop_length=self.audio_cfg["hop_length"],
                n_mels=self.audio_cfg["n_mels"],
                fmin=self.audio_cfg["fmin"],
                fmax=self.audio_cfg["fmax"],
            )
            spec = normalize_spectrogram(spec)
            # CNN için (C, H, W) = (1, n_mels, T)
            x = torch.from_numpy(spec).unsqueeze(0)
        else:
            vec = extract_feature_vector(
                y=y,
                sample_rate=self.audio_cfg["sample_rate"],
                n_mfcc=self.audio_cfg["n_mfcc"],
                n_fft=self.audio_cfg["n_fft"],
                hop_length=self.audio_cfg["hop_length"],
            )
            x = torch.from_numpy(vec)

        return x, torch.tensor(sample.label, dtype=torch.long)


def collate_pad_time(batch):
    """
    Farklı zaman uzunluklarında spektrogramları maksimum uzunluğa sıfır
    doldurarak paketler. Sabit süre kullanıldığında bile güvenli fallback.
    """
    xs, ys = zip(*batch)
    if xs[0].ndim == 3:  # CNN
        max_t = max(x.shape[-1] for x in xs)
        padded = torch.zeros(len(xs), xs[0].shape[0], xs[0].shape[1], max_t)
        for i, x in enumerate(xs):
            padded[i, :, :, : x.shape[-1]] = x
        return padded, torch.stack(ys)
    return torch.stack(xs), torch.stack(ys)
