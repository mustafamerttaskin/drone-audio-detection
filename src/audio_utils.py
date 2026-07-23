"""
audio_utils.py
----------------
Ses dosyalarını yükleme, standartlaştırma ve augmentasyon yardımcıları.

Tasarım kararları
-----------------
- librosa.load ile mono + yeniden örnekleme yapılır; tüm alt-akış tek
  örnekleme frekansı varsayar.
- Sabit uzunluk politikası: kısa örnekler sıfırla doldurulur (pad),
  uzun örnekler rastgele bir pencereden kesilir (random crop).
  Eğitimde bu, model için zamansal augmentasyon etkisi de yaratır.
- Augmentasyonlar yalnızca eğitim setine uygulanır (bkz. dataset.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: str | Path,
    sample_rate: int,
    duration_sec: float,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Ses dosyasını yükler, mono'ya indirger, hedef örnekleme frekansına
    yeniden örnekler ve sabit uzunluğa getirir.

    Parameters
    ----------
    path : str | Path
        Ses dosyasının yolu (wav, mp3, flac, ogg vb.).
    sample_rate : int
        Hedef örnekleme frekansı (Hz).
    duration_sec : float
        Çıkışın sabit uzunluğu (saniye).
    random_crop : bool
        True ise ses uzunsa rastgele pencereden kırpılır (augmentasyon).
        False ise ortadan kırpılır (deterministik, değerlendirme için).
    rng : np.random.Generator | None
        Rastgele kırpma için üreteç. None ise numpy varsayılanı kullanılır.

    Returns
    -------
    np.ndarray
        Şekli (target_len,) olan float32 ses dizisi.
    """
    if rng is None:
        rng = np.random.default_rng()

    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    target_len = int(round(sample_rate * duration_sec))

    if len(y) < target_len:
        # Sıfır doldurma (sağa)
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    elif len(y) > target_len:
        if random_crop:
            start = int(rng.integers(0, len(y) - target_len + 1))
        else:
            start = (len(y) - target_len) // 2
        y = y[start : start + target_len]

    return y.astype(np.float32)


def add_gaussian_noise(
    y: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Belirtilen SNR (dB) değerini sağlayacak şekilde beyaz gürültü ekler.

    SNR yüksek → az gürültü, SNR düşük → çok gürültü.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(y**2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)


def time_shift(
    y: np.ndarray,
    max_shift_ratio: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sinyali dairesel olarak kaydırır. Model, ses olayının pencere
    içindeki konumuna aşırı bağımlı olmasın diye kullanılır.
    """
    if rng is None:
        rng = np.random.default_rng()
    shift = int(rng.integers(-int(len(y) * max_shift_ratio),
                             int(len(y) * max_shift_ratio) + 1))
    return np.roll(y, shift).astype(np.float32)


def pitch_shift(
    y: np.ndarray,
    sample_rate: int,
    n_steps: float,
) -> np.ndarray:
    """
    n_steps yarı-ton kadar ton kaydırır. İHA motoru frekansındaki
    üretici/model varyasyonlarını simüle eder.
    """
    return librosa.effects.pitch_shift(y=y, sr=sample_rate, n_steps=n_steps).astype(np.float32)


def save_wav(path: str | Path, y: np.ndarray, sample_rate: int) -> None:
    """Yardımcı: hızlıca wav yazma (debug ve demo için)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sample_rate, subtype="PCM_16")


def compute_zero_crossing_rate(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Sıfır geçiş oranı (Zero-Crossing Rate).
    İHA sesinde ZCR belirli bir bantta yoğunlaşır → ayırt edici öznitelik.
    """
    return librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)


def compute_short_time_energy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Kısa-zamanlı enerji (Short-Time Energy).
    İHA'nın sabit dönen pervanesi → yüksek ve düşük varyanslı STE profili.
    """
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return (frames**2).sum(axis=0, keepdims=True)


def load_and_preprocess(
    path: str | Path,
    audio_cfg: dict,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Yaygın ön-işleme adımlarını tek satırda uygulayan yüksek seviyeli yardımcı.

    Returns
    -------
    (y, sr) : işlenmiş ses ve örnekleme frekansı
    """
    y = load_audio(
        path=path,
        sample_rate=audio_cfg["sample_rate"],
        duration_sec=audio_cfg["duration_sec"],
        random_crop=random_crop,
        rng=rng,
    )
    return y, audio_cfg["sample_rate"]
