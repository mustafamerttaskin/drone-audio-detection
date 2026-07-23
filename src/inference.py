"""
inference.py
------------
Eğitilmiş modelle tek bir ses dosyası üzerinde tahmin.

Kullanım (CLI):
    python -m src.inference --audio path/to/sound.wav --model cnn
    python -m src.inference --audio path/to/sound.wav --model svm

Programatik (Streamlit veya başka bir Python arayüzünden):
    from src.inference import Predictor
    p = Predictor.from_checkpoint("models/cnn_best.pt", model_type="cnn")
    result = p.predict("clip.wav")  # {'label': 'drone', 'confidence': 0.94}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from src.audio_utils import load_audio
from src.feature_extraction import (
    extract_feature_vector,
    extract_mel_spectrogram,
    normalize_spectrogram,
)
from src.models.cnn_model import DroneCNN


@dataclass
class PredictionResult:
    label: str
    label_idx: int
    confidence: float
    probabilities: dict  # {"not_drone": 0.06, "drone": 0.94}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_idx": self.label_idx,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class Predictor:
    """Ortak arayüz: hem CNN hem SVM modelini aynı API ile sunar."""

    def __init__(
        self,
        model_type: str,
        model,
        audio_cfg: dict,
        classes: list[str],
        device: torch.device | None = None,
    ):
        assert model_type in {"cnn", "svm"}
        self.model_type = model_type
        self.model = model
        self.audio_cfg = audio_cfg
        self.classes = classes
        self.device = device or torch.device("cpu")

    # -------- Yükleme -------- #
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        model_type: str,
        config_path: str | Path | None = None,
    ) -> "Predictor":
        ckpt_path = Path(ckpt_path)

        if model_type == "cnn":
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt["config"]
            model = DroneCNN(n_classes=len(cfg["classes"]))
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            return cls("cnn", model, cfg["audio"], cfg["classes"])

        # SVM
        pipe = joblib.load(ckpt_path)
        if config_path is None:
            config_path = Path("configs/config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls("svm", pipe, cfg["audio"], cfg["classes"])

    # -------- Tahmin -------- #
    def predict(self, audio_path: str | Path) -> PredictionResult:
        y = load_audio(
            path=audio_path,
            sample_rate=self.audio_cfg["sample_rate"],
            duration_sec=self.audio_cfg["duration_sec"],
            random_crop=False,
        )

        if self.model_type == "cnn":
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
            x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            vec = extract_feature_vector(
                y=y,
                sample_rate=self.audio_cfg["sample_rate"],
                n_mfcc=self.audio_cfg["n_mfcc"],
                n_fft=self.audio_cfg["n_fft"],
                hop_length=self.audio_cfg["hop_length"],
            )
            probs = self.model.predict_proba(vec.reshape(1, -1))[0]

        idx = int(np.argmax(probs))
        return PredictionResult(
            label=self.classes[idx],
            label_idx=idx,
            confidence=float(probs[idx]),
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
        )


# -------- CLI -------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Tek dosya üzerinde tahmin")
    parser.add_argument("--audio", required=True, help="Ses dosyası yolu")
    parser.add_argument("--model", choices=["cnn", "svm"], default="cnn")
    parser.add_argument("--ckpt", default=None, help="Model dosyası (opsiyonel)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = "models/cnn_best.pt" if args.model == "cnn" else "models/svm_best.joblib"

    predictor = Predictor.from_checkpoint(args.ckpt, args.model, args.config)
    result = predictor.predict(args.audio)
    print(f"Tahmin: {result.label}  (güven: {result.confidence:.3f})")
    for c, p in result.probabilities.items():
        print(f"  {c:>10s}: {p:.4f}")


if __name__ == "__main__":
    main()
