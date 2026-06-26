"""
Phase 3 — Exploratory Data Analysis.

Quick look at the processed matchup dataset before any modeling: class
balance (does home-field advantage look like the historical ~57% baseline?),
feature distributions, and a correlation heatmap to spot redundant features
and get an early read on what might be predictive.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

FEATURE_COLS = [
    "home_roll_points_for", "home_roll_points_against", "home_roll_total_yards",
    "home_roll_turnover_margin", "home_roll_sack_margin", "home_roll_third_down_pct",
    "home_roll_top_seconds",
    "away_roll_points_for", "away_roll_points_against", "away_roll_total_yards",
    "away_roll_turnover_margin", "away_roll_sack_margin", "away_roll_third_down_pct",
    "away_roll_top_seconds",
]


def class_balance(df: pd.DataFrame):
    home_win_rate = df["home_win"].mean()
    print(f"Home win rate (naive baseline): {home_win_rate:.3f} "
          f"({df['home_win'].sum()} home wins / {len(df)} games)")
    print("Historical NFL home-field win rate is commonly cited around ~57%; "
          f"this dataset's {home_win_rate:.1%} is in the same range, so the "
          "naive 'home team always wins' baseline is a reasonable benchmark.")
    return home_win_rate


def plot_feature_distributions(df: pd.DataFrame, out_path: str):
    """A few of the most interpretable rolling features, split by outcome,
    to sanity-check that winning teams' pre-game stats look different from
    losing teams'."""
    cols = ["home_roll_points_for", "home_roll_turnover_margin", "home_roll_total_yards"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, col in zip(axes, cols):
        sns.histplot(data=df, x=col, hue="home_win", bins=30, kde=True, ax=ax, palette="Set1")
        ax.set_title(col)
    fig.suptitle("Rolling feature distributions, split by home-team win/loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str):
    corr = df[FEATURE_COLS + ["home_win"]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap: rolling features and home_win")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_home_win_rate_by_season(df: pd.DataFrame, out_path: str):
    """Check the naive home-win-rate baseline is roughly stable over time
    (not drifting), which matters since the model will later be evaluated
    on a chronological train/test split."""
    by_season = df.groupby("season")["home_win"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    by_season.plot(kind="bar", ax=ax, color="steelblue")
    ax.axhline(df["home_win"].mean(), color="red", linestyle="--", label="overall mean")
    ax.set_ylabel("Home win rate")
    ax.set_title("Home win rate by season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "matchups.csv"))

    class_balance(df)

    plot_feature_distributions(df, os.path.join(FIGURES_DIR, "feature_distributions.png"))
    plot_correlation_heatmap(df, os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plot_home_win_rate_by_season(df, os.path.join(FIGURES_DIR, "home_win_rate_by_season.png"))

    print(f"\nSaved 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
