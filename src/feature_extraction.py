"""
feature_extraction.py
---------------------
İki farklı öznitelik gösterimi üretir:

1) `extract_mel_spectrogram`  → CNN girişi (2B, log-mel spektrogram).
2) `extract_feature_vector`   → Klasik ML (SVM/RandomForest) için 1B özet
   vektör. MFCC + delta + delta² + spektral centroid/bandwidth/rolloff/
   contrast + ZCR + RMS istatistikleri (mean & std) birleştirilir.

Neden iki yol var?
------------------
CNN, uzamsal-spektral örüntüleri öğrenebiliyor ama yorumlanması zor ve
küçük veri setinde overfit'e yatkın. SVM baseline ise hem hızlı hem de
raporda "temel yöntem vs derin öğrenme" karşılaştırması yapılmasını sağlar.
Bu, endüstriyel raporlarda beklenen bir ablation çalışmasıdır.
"""

from __future__ import annotations

import librosa
import numpy as np


# --------------------------------------------------------------------- #
# CNN girişi: log-mel spektrogram                                        #
# --------------------------------------------------------------------- #
def extract_mel_spectrogram(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Log-ölçekli mel-spektrogram üretir. Dönen dizi (n_mels, T) şeklindedir
    ve CNN'e "tek kanallı imge" olarak beslenir.
    """
    if fmax is None:
        fmax = sample_rate / 2

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Per-örnek z-score normalizasyonu. Model, mutlak dB seviyesine değil
    örüntüye odaklanır.
    """
    mean = spec.mean()
    std = spec.std() + 1e-6
    return ((spec - mean) / std).astype(np.float32)


# --------------------------------------------------------------------- #
# Klasik ML için 1B öznitelik vektörü                                    #
# --------------------------------------------------------------------- #
def _stats(x: np.ndarray) -> np.ndarray:
    """Her satır (özellik) için ortalama ve std → 1B vektöre yığar."""
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


def extract_feature_vector(
    y: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Aşağıdaki özniteliklerin frame-level ortalama ve standart sapmalarını
    birleştirerek tek bir öznitelik vektörü üretir:

    - MFCC (n_mfcc katsayı)
    - MFCC delta (birinci türev)
    - MFCC delta² (ikinci türev)
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff (0.85)
    - Spectral contrast (7 bant)
    - Zero-crossing rate
    - RMS enerji

    Toplam boyut deterministiktir; SVM için yeterince ayırt edicidir.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_d1 = librosa.feature.delta(mfcc, order=1)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

    parts = [
        _stats(mfcc),
        _stats(mfcc_d1),
        _stats(mfcc_d2),
        _stats(centroid),
        _stats(bandwidth),
        _stats(rolloff),
        _stats(contrast),
        _stats(zcr),
        _stats(rms),
    ]
    return np.concatenate(parts).astype(np.float32)


def feature_vector_size(n_mfcc: int) -> int:
    """
    Vektör boyutunun deterministik hesaplanışı.
    3*n_mfcc (MFCC + Δ + Δ²) + 1 (centroid) + 1 (bandwidth)
    + 1 (rolloff) + 7 (contrast) + 1 (zcr) + 1 (rms) — her biri (mean, std)
    olduğu için toplam 2 katına çıkar.
    """
    single = 3 * n_mfcc + 1 + 1 + 1 + 7 + 1 + 1
    return single * 2
