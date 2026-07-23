"""
tests/test_features.py
----------------------
Sığ ama işlevsel testler. Amaç, feature çıkarımının deterministik olması,
beklenen boyutları döndürmesi ve augmentasyonun sinyali bozmaması.

Çalıştırma:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio_utils import add_gaussian_noise, load_audio, time_shift
from src.feature_extraction import (
    extract_feature_vector,
    extract_mel_spectrogram,
    feature_vector_size,
)


SR = 22050
DUR = 2.0


def _make_signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * DUR)) / SR
    y = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.standard_normal(t.size)
    return y.astype(np.float32)


def test_mel_spectrogram_shape():
    y = _make_signal()
    spec = extract_mel_spectrogram(
        y=y, sample_rate=SR, n_fft=2048, hop_length=512, n_mels=64
    )
    assert spec.ndim == 2
    assert spec.shape[0] == 64  # n_mels
    assert spec.shape[1] > 0


def test_feature_vector_size_matches_formula():
    y = _make_signal()
    n_mfcc = 40
    vec = extract_feature_vector(
        y=y, sample_rate=SR, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
    )
    assert vec.ndim == 1
    assert vec.shape[0] == feature_vector_size(n_mfcc)


def test_feature_vector_deterministic():
    y = _make_signal(seed=7)
    v1 = extract_feature_vector(y, SR, 40, 2048, 512)
    v2 = extract_feature_vector(y, SR, 40, 2048, 512)
    assert np.allclose(v1, v2)


def test_noise_augmentation_preserves_length():
    y = _make_signal()
    y_noisy = add_gaussian_noise(y, snr_db=15.0, rng=np.random.default_rng(1))
    assert y_noisy.shape == y.shape
    assert not np.allclose(y_noisy, y)


def test_time_shift_preserves_length():
    y = _make_signal()
    y_shift = time_shift(y, max_shift_ratio=0.2, rng=np.random.default_rng(1))
    assert y_shift.shape == y.shape
