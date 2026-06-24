"""
Phase 4 — Modeling.

Chronological train/test split, not random: a random split would let the
model train on games from December 2023 and be tested on games from
January 2023 -- effectively peeking at the future relative to some test
games, which isn't how this model would ever actually be used (predicting
a game before it happens, using only data from before it happens). We
train on 2010-2021 and test on 2022-2023, which simulates real predictive
use: at prediction time, only the past is available.

Hyperparameter tuning uses stratified k-fold CV *within the training set
only*. The chronological test set is touched exactly once, at the very end
(in evaluate.py), so its performance number is a single honest estimate of
out-of-sample accuracy -- not something we tuned against.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TEST_SEASONS_FROM = 2022  # train on seasons before this, test on this and later

FEATURE_COLS = [
    "home_roll_points_for", "home_roll_points_against", "home_roll_total_yards",
    "home_roll_turnover_margin", "home_roll_sack_margin", "home_roll_third_down_pct",
    "home_roll_top_seconds",
    "away_roll_points_for", "away_roll_points_against", "away_roll_total_yards",
    "away_roll_turnover_margin", "away_roll_sack_margin", "away_roll_third_down_pct",
    "away_roll_top_seconds",
    "home_rest", "away_rest",
]
LABEL_COL = "home_win"


def chronological_split(df: pd.DataFrame):
    train = df[df["season"] < TEST_SEASONS_FROM].reset_index(drop=True)
    test = df[df["season"] >= TEST_SEASONS_FROM].reset_index(drop=True)
    print(f"Train: {len(train)} games (seasons {train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test)} games (seasons {test['season'].min()}-{test['season'].max()})")
    return train, test


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    """Fit on train only -- scaling using test-set statistics would leak
    information about the test distribution into preprocessing."""
    scaler = StandardScaler()
    scaler.fit(train[FEATURE_COLS])
    return scaler


def tune_model(name, estimator, param_grid, X_train, y_train, cv):
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  {name}: best params {search.best_params_}, "
          f"best CV accuracy {search.best_score_:.4f}")
    return search.best_estimator_


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "matchups.csv"))
    train, test = chronological_split(df)

    scaler = fit_scaler(train)
    X_train = scaler.transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])
    y_train = train[LABEL_COL].values
    y_test = test[LABEL_COL].values

    # Stratified k-fold for model selection/tuning on the training set only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTuning models via 5-fold stratified CV on the training set...")

    models = {}

    models["logistic_regression"] = tune_model(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1.0, 10.0]},
        X_train, y_train, cv,
    )

    models["decision_tree"] = tune_model(
        "DecisionTree",
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]},
        X_train, y_train, cv,
    )

    models["random_forest"] = tune_model(
        "RandomForest",
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 300], "max_depth": [5, 10, None]},
        X_train, y_train, cv,
    )

    print("\nFitting final models on the full training set (best params from CV)...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  {name}: train acc {train_acc:.4f}, held-out test acc {test_acc:.4f}")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.joblib"))

    # Also persist the chronological test set predictions inputs for evaluate.py
    test.to_csv(os.path.join(PROCESSED_DIR, "test_set.csv"), index=False)
    train.to_csv(os.path.join(PROCESSED_DIR, "train_set.csv"), index=False)

    print(f"\nSaved {len(models)} models + scaler to {MODELS_DIR}")


if __name__ == "__main__":
    main()
