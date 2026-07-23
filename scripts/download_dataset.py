"""
scripts/download_dataset.py
---------------------------
Ücretsiz veri seti kaynaklarını listeler ve klasör yapısını hazırlar.

Bu proje herkese açık iki veri setini önerir:

1) Sara Al-Emadi Drone Audio Dataset (yes-drone / no-drone)
   https://github.com/saraalemadi/DroneAudioDataset
   - Klonlayın: git clone https://github.com/saraalemadi/DroneAudioDataset.git
   - `Binary_Drone_Audio/yes_drone/*.wav`   → data/raw/drone/
   - `Binary_Drone_Audio/unknown/*.wav`     → data/raw/not_drone/

2) ESC-50 (çevresel ses referansı, "not_drone" için ek çeşitlilik)
   https://github.com/karolpiczak/ESC-50
   - `helicopter`, `airplane`, `wind`, `rain` gibi sınıfları
     `data/raw/not_drone/` klasörüne kopyalayabilirsiniz.

Not: Ticari veri seti indirme yerine bu iki repo tamamen ücretsiz ve
akademik/kişisel kullanım için serbesttir.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"


def prepare_folders() -> None:
    for cls in ("drone", "not_drone"):
        (DATA_ROOT / cls).mkdir(parents=True, exist_ok=True)
        keep = DATA_ROOT / cls / ".gitkeep"
        keep.touch(exist_ok=True)
    print(f"[OK] Klasörler hazır: {DATA_ROOT}/drone  ve  {DATA_ROOT}/not_drone")


def print_instructions() -> None:
    print(
        """
    ============================================================
      VERİ SETİ TALİMATLARI
    ============================================================

    Bu proje aşağıdaki ücretsiz veri setlerini önerir:

    1) Sara Al-Emadi Drone Audio Dataset  (birincil kaynak)
       git clone https://github.com/saraalemadi/DroneAudioDataset.git

       Ardından:
       - Binary_Drone_Audio/yes_drone/*.wav  →  data/raw/drone/
       - Binary_Drone_Audio/unknown/*.wav    →  data/raw/not_drone/

    2) ESC-50  (opsiyonel, negatif sınıf çeşitliliği için)
       git clone https://github.com/karolpiczak/ESC-50.git
       Meta CSV'sinden helikopter/uçak/rüzgâr sınıflarını seçip
       data/raw/not_drone/ altına kopyalayın.

    Sonrasında:
       python -m src.train --model svm
       python -m src.train --model cnn
       streamlit run app/streamlit_app.py
    ============================================================
    """
    )


if __name__ == "__main__":
    prepare_folders()
    print_instructions()
