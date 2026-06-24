"""
Phase 6 (optional stretch) — Single hypothetical matchup demo.

For the live presentation: predicts a win probability for a hypothetical
home_team vs away_team matchup, using each team's most recent rolling
stats available in the processed dataset as a stand-in for "current form".
This is a convenience script for demoing the trained model interactively,
not part of the modeling pipeline itself.

Usage:
    python src/predict_matchup.py HOME_TEAM AWAY_TEAM
    python src/predict_matchup.py KC SF
"""

import os
import sys

import joblib
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Generic stat names (no home_/away_ prefix) used to describe one team's form.
STAT_NAMES = [
    "roll_points_for", "roll_points_against", "roll_total_yards",
    "roll_turnover_margin", "roll_sack_margin", "roll_third_down_pct", "roll_top_seconds",
]
DEFAULT_REST_DAYS = 7  # standard week-to-week rest; used since the demo doesn't ask for a real date


def latest_team_form(matchups: pd.DataFrame, team: str) -> pd.Series:
    """Find this team's most recent game (as either home or away) and pull
    out their rolling stats from that row, relabeled to generic STAT_NAMES."""
    as_home = matchups[matchups["home_team"] == team].assign(side="home")
    as_away = matchups[matchups["away_team"] == team].assign(side="away")
    appearances = pd.concat([as_home, as_away]).sort_values("gameday")

    if appearances.empty:
        raise ValueError(f"No games found for team '{team}'. Check the abbreviation.")

    latest = appearances.iloc[-1]
    prefix = "home_" if latest["side"] == "home" else "away_"
    form = pd.Series({stat: latest[f"{prefix}{stat}"] for stat in STAT_NAMES})
    form["as_of_game"] = latest["game_id"]
    return form


def build_feature_row(home_team: str, away_team: str, matchups: pd.DataFrame, feature_cols) -> pd.DataFrame:
    home_form = latest_team_form(matchups, home_team)
    away_form = latest_team_form(matchups, away_team)

    print(f"Using {home_team}'s form as of {home_form['as_of_game']}")
    print(f"Using {away_team}'s form as of {away_form['as_of_game']}")

    row = {}
    for stat in STAT_NAMES:
        row[f"home_{stat}"] = home_form[stat]
        row[f"away_{stat}"] = away_form[stat]
    row["home_rest"] = DEFAULT_REST_DAYS
    row["away_rest"] = DEFAULT_REST_DAYS

    return pd.DataFrame([row])[feature_cols]


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/predict_matchup.py HOME_TEAM AWAY_TEAM")
        print("Example: python src/predict_matchup.py KC SF")
        sys.exit(1)

    home_team, away_team = sys.argv[1].upper(), sys.argv[2].upper()

    matchups = pd.read_csv(os.path.join(PROCESSED_DIR, "matchups.csv"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    X = build_feature_row(home_team, away_team, matchups, feature_cols)
    X_scaled = scaler.transform(X)
    home_win_prob = model.predict_proba(X_scaled)[0, 1]

    print(f"\n{home_team} (home) vs {away_team} (away)")
    print(f"Predicted probability {home_team} wins: {home_win_prob:.1%}")
    print(f"Predicted probability {away_team} wins: {1 - home_win_prob:.1%}")


if __name__ == "__main__":
    main()
