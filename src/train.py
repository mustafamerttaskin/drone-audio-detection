"""
train.py
--------
Uçtan uca eğitim scripti. CLI:

    python -m src.train --model cnn --config configs/config.yaml
    python -m src.train --model svm --config configs/config.yaml

Çıktılar:
- models/cnn_best.pt veya models/svm_best.joblib
- reports/<model>/metrics.json, confusion_matrix.png, roc_curve.png,
  training_curves.png (yalnızca CNN)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    DroneAudioDataset,
    collate_pad_time,
    scan_dataset,
    stratified_split,
)
from src.models.cnn_model import DroneCNN
from src.models.svm_baseline import SVMConfig, build_svm_pipeline


# ---------------------------------------------------------------- #
# Yardımcılar                                                       #
# ---------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- #
# Değerlendirme                                                     #
# ---------------------------------------------------------------- #
@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray  # positive class olasılığı


def compute_metrics(y_true, y_pred, y_score) -> EvalResult:
    return EvalResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
    )


def save_reports(result: EvalResult, out_dir: Path, class_names: list[str], tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics = {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # classification report (text)
    report_text = classification_report(
        result.y_true, result.y_pred, target_names=class_names, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    # confusion matrix
    cm = confusion_matrix(result.y_true, result.y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    plt.title(f"Confusion Matrix — {tag}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    # ROC curve
    if not np.isnan(result.roc_auc):
        fpr, tpr, _ = roc_curve(result.y_true, result.y_score)
        plt.figure(figsize=(4.5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {result.roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {tag}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=140)
        plt.close()


# ---------------------------------------------------------------- #
# CNN eğitimi                                                       #
# ---------------------------------------------------------------- #
def train_cnn(cfg: dict) -> None:
    set_seed(cfg["split"]["random_state"])
    device = resolve_device(cfg["training"]["device"])
    print(f"[CNN] Cihaz: {device}")

    # Veri
    samples = scan_dataset(cfg["paths"]["data_root"], cfg["classes"])
    train_s, val_s, test_s = stratified_split(
        samples,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    print(f"[CNN] Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    train_ds = DroneAudioDataset(
        train_s, cfg["audio"], feature_type="mel",
        training=True, aug_cfg=cfg["augmentation"],
        seed=cfg["split"]["random_state"],
    )
    val_ds = DroneAudioDataset(
        val_s, cfg["audio"], feature_type="mel", training=False,
    )
    test_ds = DroneAudioDataset(
        test_s, cfg["audio"], feature_type="mel", training=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=collate_pad_time, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        collate_fn=collate_pad_time, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        collate_fn=collate_pad_time, num_workers=0,
    )

    # Model
    model = DroneCNN(n_classes=len(cfg["classes"])).to(device)
    print(f"[CNN] Parametre sayısı: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Eğitim döngüsü
    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    stale = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "cnn_best.pt"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch:02d} - train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= max(n, 1)

        # Val
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_loss /= max(total, 1)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        print(
            f"[CNN] Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            stale = 0
            torch.save({"model_state": model.state_dict(),
                        "config": cfg}, ckpt_path)
            print(f"[CNN]   ↳ En iyi model kaydedildi: {ckpt_path}")
        else:
            stale += 1
            if stale >= patience:
                print(f"[CNN] Early stopping (sabır={patience}).")
                break

    # Eğitim eğrileri
    reports_dir = Path(cfg["paths"]["reports_dir"]) / "cnn"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history["val_acc"], label="val_acc", color="green")
    ax[1].set_title("Validation Accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "training_curves.png", dpi=140)
    plt.close()

    # Test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs[:, 1].tolist())

    result = compute_metrics(y_true, y_pred, y_score)
    print(f"\n[CNN] TEST | acc={result.accuracy:.4f}  f1={result.f1:.4f}  "
          f"auc={result.roc_auc:.4f}")
    save_reports(result, reports_dir, cfg["classes"], tag="CNN")


# ---------------------------------------------------------------- #
# SVM eğitimi                                                       #
# ---------------------------------------------------------------- #
def _dataset_to_arrays(ds: DroneAudioDataset):
    X, y = [], []
    for i in range(len(ds)):
        xb, yb = ds[i]
        X.append(xb.numpy())
        y.append(int(yb))
    return np.stack(X), np.array(y)


def train_svm(cfg: dict) -> None:
    set_seed(cfg["split"]["random_state"])
    samples = scan_dataset(cfg["paths"]["data_root"], cfg["classes"])
    train_s, val_s, test_s = stratified_split(
        samples,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    print(f"[SVM] Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    # SVM için augmentasyon kapalı; tekrarlanabilir öznitelikler
    train_ds = DroneAudioDataset(train_s, cfg["audio"], feature_type="vector", training=False)
    val_ds   = DroneAudioDataset(val_s,   cfg["audio"], feature_type="vector", training=False)
    test_ds  = DroneAudioDataset(test_s,  cfg["audio"], feature_type="vector", training=False)

    print("[SVM] Öznitelik çıkarımı...")
    t0 = time.time()
    X_train, y_train = _dataset_to_arrays(train_ds)
    X_val,   y_val   = _dataset_to_arrays(val_ds)
    X_test,  y_test  = _dataset_to_arrays(test_ds)
    print(f"[SVM] Süre: {time.time() - t0:.1f}s | Vektör boyutu: {X_train.shape[1]}")

    # Train (train + val birleşimi)
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    pipe = build_svm_pipeline(SVMConfig())
    pipe.fit(X_full, y_full)

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "svm_best.joblib")

    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    result = compute_metrics(y_test, y_pred, y_score)
    print(f"\n[SVM] TEST | acc={result.accuracy:.4f}  f1={result.f1:.4f}  "
          f"auc={result.roc_auc:.4f}")

    reports_dir = Path(cfg["paths"]["reports_dir"]) / "svm"
    save_reports(result, reports_dir, cfg["classes"], tag="SVM")


# ---------------------------------------------------------------- #
# CLI                                                               #
# ---------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Drone audio detection eğitici")
    parser.add_argument("--model", choices=["cnn", "svm"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model == "cnn":
        train_cnn(cfg)
    else:
        train_svm(cfg)


if __name__ == "__main__":
    main()
