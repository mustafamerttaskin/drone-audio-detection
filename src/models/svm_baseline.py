"""
svm_baseline.py
---------------
Klasik ML baseline'ı. StandardScaler + RBF-SVM pipeline'ı, MFCC tabanlı
1B öznitelik vektörleri üzerinde eğitilir.

Bu baseline neden önemli?
- Küçük veri setinde CNN'i geride bırakabilir → dürüst raporlama.
- Endüstriyel raporlarda "önce basit yöntem, sonra derin öğrenme" akışı
  standarttır. Ablation için gereklidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class SVMConfig:
    C: float = 10.0
    gamma: str | float = "scale"
    kernel: str = "rbf"
    probability: bool = True
    class_weight: str | None = "balanced"
    random_state: int = 42


def build_svm_pipeline(cfg: SVMConfig | None = None) -> Pipeline:
    """Ölçekleme + RBF-SVM pipeline'ı üretir."""
    if cfg is None:
        cfg = SVMConfig()
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=cfg.C,
                    kernel=cfg.kernel,
                    gamma=cfg.gamma,
                    probability=cfg.probability,
                    class_weight=cfg.class_weight,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Sınıf olasılıklarını döndürür (positive class 'drone' için sütun 1)."""
    return pipeline.predict_proba(X)
