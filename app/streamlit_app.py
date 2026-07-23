"""
Streamlit demo — canlı arayüz.
Çalıştırma:
    streamlit run app/streamlit_app.py

Kullanıcı bir ses dosyası (wav/mp3/ogg/flac) yükler; uygulama
tahmin sonucunu, sınıf olasılıklarını ve mel-spektrogramı gösterir.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

# src paketini bulunabilir yap
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_utils import load_audio  # noqa: E402
from src.feature_extraction import extract_mel_spectrogram, normalize_spectrogram  # noqa: E402
from src.inference import Predictor  # noqa: E402


st.set_page_config(
    page_title="Drone Audio Detection",
    page_icon="🛩️",
    layout="centered",
)


# --- Model yükleme (cache'li) ---
@st.cache_resource
def load_predictor(model_type: str) -> Predictor | None:
    ckpt = ROOT / "models" / ("cnn_best.pt" if model_type == "cnn" else "svm_best.joblib")
    if not ckpt.exists():
        return None
    return Predictor.from_checkpoint(ckpt, model_type)


# --- Başlık ---
st.title("🛩️ Drone Audio Detection")
st.caption(
    "Akustik imzadan İHA tespiti — MFCC/log-mel öznitelikleri üzerinde "
    "eğitilmiş SVM ve CNN modelleri."
)

with st.sidebar:
    st.header("Ayarlar")
    model_type = st.radio("Model", ["cnn", "svm"], index=0,
                          help="CNN log-mel spektrogram üzerinde, SVM klasik MFCC özniteliği üzerinde çalışır.")
    st.markdown("---")
    st.markdown(
        "**Nasıl kullanılır?**\n"
        "1. Bir ses dosyası yükleyin (wav/mp3/ogg/flac).\n"
        "2. Modelin `models/` klasöründe eğitilmiş olduğundan emin olun.\n"
        "3. Sonuç ve mel-spektrogram aşağıda gösterilir."
    )

predictor = load_predictor(model_type)

if predictor is None:
    st.error(
        f"Eğitilmiş `{model_type}` modeli bulunamadı. "
        f"Önce `python -m src.train --model {model_type}` komutunu çalıştırın."
    )
    st.stop()

# --- Dosya yükleme ---
uploaded = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Analize başlamak için bir ses dosyası yükleyin.")
    st.stop()

# Geçici olarak diske yaz (librosa çoğu formatı yoldan okumayı tercih eder)
tmp_path = ROOT / "app" / "_tmp_upload"
tmp_path.mkdir(exist_ok=True)
tmp_file = tmp_path / uploaded.name
with open(tmp_file, "wb") as f:
    f.write(uploaded.getbuffer())

st.audio(uploaded)

# --- Tahmin ---
with st.spinner("Analiz ediliyor..."):
    result = predictor.predict(tmp_file)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric(
        label="Tahmin",
        value=result.label.upper(),
        delta=f"{result.confidence * 100:.1f}% güven",
    )
with col2:
    st.metric(label="Model", value=model_type.upper())

st.subheader("Sınıf olasılıkları")
prob_data = {"Sınıf": list(result.probabilities.keys()),
             "Olasılık": list(result.probabilities.values())}
st.bar_chart(prob_data, x="Sınıf", y="Olasılık")

# --- Mel-spektrogram ---
st.subheader("Log-Mel Spektrogram")
audio_cfg = predictor.audio_cfg
y = load_audio(
    tmp_file,
    sample_rate=audio_cfg["sample_rate"],
    duration_sec=audio_cfg["duration_sec"],
    random_crop=False,
)
spec = extract_mel_spectrogram(
    y=y,
    sample_rate=audio_cfg["sample_rate"],
    n_fft=audio_cfg["n_fft"],
    hop_length=audio_cfg["hop_length"],
    n_mels=audio_cfg["n_mels"],
    fmin=audio_cfg["fmin"],
    fmax=audio_cfg["fmax"],
)
fig, ax = plt.subplots(figsize=(8, 3))
img = librosa.display.specshow(
    spec, sr=audio_cfg["sample_rate"], hop_length=audio_cfg["hop_length"],
    x_axis="time", y_axis="mel", fmin=audio_cfg["fmin"],
    fmax=audio_cfg["fmax"], ax=ax,
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("Log-Mel Spectrogram")
st.pyplot(fig)

st.markdown("---")
st.caption(
    "Not: Bu bir öğrenci projesidir. Model performansı eğitim veri setinin "
    "kalitesine ve çeşitliliğine bağlıdır. Gerçek dünyada rüzgâr, uçak, "
    "trafik gibi karıştırıcı sesler için ek fine-tuning gereklidir."
)

# Temizlik
try:
    os.remove(tmp_file)
except OSError:
    pass
