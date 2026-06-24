"""
Phase 5 — Evaluation.

This is the one place the chronological test set (2022-2023, untouched
since train_models.py split it off) gets used. Every number here is a
single, honest out-of-sample estimate -- nothing here was used to pick
hyperparameters.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]
LABEL_COL = "home_win"


def load_test_data():
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_set.csv"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    X_test = scaler.transform(test[feature_cols])
    y_test = test[LABEL_COL].values
    return test, X_test, y_test, feature_cols


def naive_baseline_metrics(y_test):
    """Home team always wins -- the benchmark every model needs to beat."""
    y_pred = [1] * len(y_test)
    return {
        "model": "naive_home_always_wins",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": float("nan"),  # undefined for a constant prediction
    }


def model_metrics(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def plot_roc_curves(models, X_test, y_test, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_title("ROC curves (chronological test set, 2022-2023)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(models, X_test, y_test, out_path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["away win", "home win"], ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importances(model, feature_cols, out_path):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    test, X_test, y_test, feature_cols = load_test_data()

    models = {name: joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib")) for name in MODEL_NAMES}

    rows = [naive_baseline_metrics(y_test)]
    for name, model in models.items():
        rows.append(model_metrics(name, model, X_test, y_test))

    results = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Results on held-out chronological test set (2022-2023):\n")
    print(results)

    results.to_csv(os.path.join(PROCESSED_DIR, "results_table.csv"))

    plot_roc_curves(models, X_test, y_test, os.path.join(FIGURES_DIR, "roc_curves.png"))
    plot_confusion_matrices(models, X_test, y_test, os.path.join(FIGURES_DIR, "confusion_matrices.png"))
    plot_feature_importances(
        models["random_forest"], feature_cols, os.path.join(FIGURES_DIR, "feature_importances.png")
    )

    best_model = results.drop(index="naive_home_always_wins")["accuracy"].idxmax()
    best_acc = results.loc[best_model, "accuracy"]
    baseline_acc = results.loc["naive_home_always_wins", "accuracy"]
    beats = "beats" if best_acc > baseline_acc else "does not beat"
    print(f"\nBest model ({best_model}, {best_acc:.3f} accuracy) {beats} "
          f"the naive home-team-always-wins baseline ({baseline_acc:.3f}).")

    print(f"\nSaved 3 figures to {FIGURES_DIR} and results table to data/processed/results_table.csv")


if __name__ == "__main__":
    main()
