"""
train.py  —  Model Training & Evaluation Pipeline
MMAI1002 Final Project: Crop Recommendation for Aceh/Sumatra
--------------------------------------------------------------
Trains Decision Tree, Random Forest, and KNN classifiers.
Produces all evaluation artefacts needed for Section 5 of the manuscript.

Usage:
    python train.py
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE
FIG_DIR  = BASE / "figures"
MDL_DIR  = BASE / "models"
FIG_DIR.mkdir(exist_ok=True)
MDL_DIR.mkdir(exist_ok=True)

SEED     = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & split data
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} samples, {df['label'].nunique()} classes")

    X = df[FEATURES].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Stratified 70 / 15 / 15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_enc, test_size=0.30, random_state=SEED, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    log.info(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return (X_train_s, X_val_s, X_test_s,
            y_train, y_val, y_test,
            scaler, le)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train models
# ─────────────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=4, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=None,
            min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5, metric="euclidean", n_jobs=-1
        ),
    }


def train_and_validate(models, X_train, X_val, y_train, y_val) -> dict:
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred  = model.predict(X_val)
        val_acc   = accuracy_score(y_val, val_pred)
        val_f1    = f1_score(y_val, val_pred, average="weighted")
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "model":   model,
            "val_acc": val_acc,
            "val_f1":  val_f1,
            "cv_mean": cv_scores.mean(),
            "cv_std":  cv_scores.std(),
        }
        log.info(
            f"{name:20s} | val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation figures
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: dict):
    names    = list(results.keys())
    val_accs = [results[n]["val_acc"] for n in names]
    cv_means = [results[n]["cv_mean"] for n in names]
    cv_stds  = [results[n]["cv_std"]  for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, val_accs, w, label="Validation accuracy",
                   color="#5DCAA5", edgecolor="white")
    bars2 = ax.bar(x + w/2, cv_means, w, yerr=cv_stds, capsize=5,
                   label="5-fold CV accuracy (±std)", color="#7F77DD",
                   edgecolor="white")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 6 — Model comparison: validation & cross-validation accuracy",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_model_comparison.png", dpi=200)
    plt.close()
    log.info("Saved: fig6_model_comparison.png")


def plot_confusion_matrix(model, X_test, y_test, class_names, model_name: str):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="YlGnBu")
    ax.set_title(f"Figure 7 — Confusion matrix ({model_name})\non test set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = FIG_DIR / "fig7_confusion_matrix.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    log.info(f"Saved: fig7_confusion_matrix.png")
    return y_pred


def plot_feature_importance(rf_model, feature_names: list):
    importances = rf_model.feature_importances_
    std         = np.std([t.feature_importances_ for t in rf_model.estimators_], axis=0)
    idx         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(feature_names)),
           importances[idx], yerr=std[idx],
           color="#1D9E75", edgecolor="white", capsize=4)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=20)
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_title("Figure 8 — Feature importances (Random Forest)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_feature_importance.png", dpi=200)
    plt.close()
    log.info("Saved: fig8_feature_importance.png")


def plot_per_class_f1(y_test, y_pred, class_names):
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   output_dict=True)
    f1s    = [report[c]["f1-score"] for c in class_names]
    precs  = [report[c]["precision"] for c in class_names]
    recs   = [report[c]["recall"] for c in class_names]

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,   precs, w, label="Precision", color="#5DCAA5", edgecolor="white")
    ax.bar(x,       recs,  w, label="Recall",    color="#7F77DD", edgecolor="white")
    ax.bar(x + w,   f1s,   w, label="F1-score",  color="#D85A30", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Figure 9 — Per-class precision, recall, and F1-score (best model)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_per_class_metrics.png", dpi=200)
    plt.close()
    log.info("Saved: fig9_per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artefacts(best_name, best_model, scaler, le, results):
    joblib.dump(best_model, MDL_DIR / "crop_model.pkl")
    joblib.dump(scaler,     MDL_DIR / "scaler.pkl")
    joblib.dump(le,         MDL_DIR / "label_encoder.pkl")

    summary = {
        name: {
            "val_acc": round(r["val_acc"], 5),
            "val_f1":  round(r["val_f1"],  5),
            "cv_mean": round(r["cv_mean"], 5),
            "cv_std":  round(r["cv_std"],  5),
        }
        for name, r in results.items()
    }
    with open(MDL_DIR / "results_summary.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": summary}, f, indent=2)

    log.info(f"Artefacts saved to: {MDL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, le) = load_and_split(DATA_DIR / "aceh_crop_data.csv")

    class_names = le.classes_
    log.info(f"Classes: {list(class_names)}")

    models  = build_models()
    results = train_and_validate(models, X_train, X_val, y_train, y_val)

    # Pick best by validation accuracy
    best_name  = max(results, key=lambda k: results[k]["val_acc"])
    best_model = results[best_name]["model"]
    log.info(f"\nBest model -> {best_name}")

    # Test-set evaluation
    y_pred    = best_model.predict(X_test)
    test_acc  = accuracy_score(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred, average="weighted")
    log.info(f"Test accuracy : {test_acc:.4f}")
    log.info(f"Test F1 (wtd) : {test_f1:.4f}")
    print("\n-- Classification Report --")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Generate all figures
    plot_model_comparison(results)
    y_pred_plot = plot_confusion_matrix(best_model, X_test, y_test,
                                        class_names, best_name)

    # Feature importance (use Random Forest regardless of best)
    rf = results["Random Forest"]["model"]
    plot_feature_importance(rf, FEATURES)
    plot_per_class_f1(y_test, y_pred, class_names)

    # Save
    save_artefacts(best_name, best_model, scaler, le, results)

    log.info("Pipeline complete.")
