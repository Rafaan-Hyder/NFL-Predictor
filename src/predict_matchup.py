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


def latest_team_form(matchups: pd.DataFrame, team: str, valid_teams: set) -> pd.Series:
    """Find this team's most recent game (as either home or away) and pull
    out their rolling stats from that row, relabeled to generic STAT_NAMES."""
    if team not in valid_teams:
        raise ValueError(
            f"Unknown team code '{team}'. Valid team codes: {', '.join(sorted(valid_teams))}"
        )

    as_home = matchups[matchups["home_team"] == team].assign(side="home")
    as_away = matchups[matchups["away_team"] == team].assign(side="away")
    appearances = pd.concat([as_home, as_away]).sort_values("gameday")

    if appearances.empty:
        raise ValueError(f"No games found for team '{team}' in the dataset.")

    latest = appearances.iloc[-1]
    prefix = "home_" if latest["side"] == "home" else "away_"
    form = pd.Series({stat: latest[f"{prefix}{stat}"] for stat in STAT_NAMES})
    form["as_of_game"] = latest["game_id"]
    return form


def build_feature_row(home_team: str, away_team: str, matchups: pd.DataFrame, feature_cols, valid_teams: set) -> pd.DataFrame:
    home_form = latest_team_form(matchups, home_team, valid_teams)
    away_form = latest_team_form(matchups, away_team, valid_teams)

    print(f"Using {home_team}'s form as of {home_form['as_of_game']}")
    print(f"Using {away_team}'s form as of {away_form['as_of_game']}")

    row = {}
    for stat in STAT_NAMES:
        row[f"home_{stat}"] = home_form[stat]
        row[f"away_{stat}"] = away_form[stat]
    row["home_rest"] = DEFAULT_REST_DAYS
    row["away_rest"] = DEFAULT_REST_DAYS

    return pd.DataFrame([row])[feature_cols]


def usage_and_exit():
    print("Usage: python src/predict_matchup.py HOME_TEAM AWAY_TEAM")
    print("Example: python src/predict_matchup.py KC SF")
    sys.exit(1)


def load_artifacts():
    """Load the processed dataset and trained model, with a clear message
    if the pipeline hasn't been run yet (these files are normally already
    committed, but a fresh clone without them shouldn't crash with a raw
    FileNotFoundError)."""
    required = {
        "matchups": os.path.join(PROCESSED_DIR, "matchups.csv"),
        "scaler": os.path.join(MODELS_DIR, "scaler.joblib"),
        "feature_cols": os.path.join(MODELS_DIR, "feature_cols.joblib"),
        "model": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    }
    missing = [path for path in required.values() if not os.path.exists(path)]
    if missing:
        print("Missing required file(s):")
        for path in missing:
            print(f"  {path}")
        print("\nRun the pipeline first: python src/build_features.py && python src/train_models.py")
        sys.exit(1)

    matchups = pd.read_csv(required["matchups"])
    scaler = joblib.load(required["scaler"])
    feature_cols = joblib.load(required["feature_cols"])
    model = joblib.load(required["model"])
    return matchups, scaler, feature_cols, model


def main():
    if len(sys.argv) != 3:
        print(f"Error: expected 2 arguments (home team, away team), got {len(sys.argv) - 1}.\n")
        usage_and_exit()

    home_team, away_team = sys.argv[1].strip().upper(), sys.argv[2].strip().upper()

    if not home_team or not away_team:
        print("Error: team codes cannot be blank.\n")
        usage_and_exit()

    if home_team == away_team:
        print(f"Error: home team and away team can't both be '{home_team}'.")
        sys.exit(1)

    matchups, scaler, feature_cols, model = load_artifacts()
    valid_teams = set(matchups["home_team"]) | set(matchups["away_team"])

    try:
        X = build_feature_row(home_team, away_team, matchups, feature_cols, valid_teams)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    X_scaled = scaler.transform(X)
    home_win_prob = model.predict_proba(X_scaled)[0, 1]

    print(f"\n{home_team} (home) vs {away_team} (away)")
    print(f"Predicted probability {home_team} wins: {home_win_prob:.1%}")
    print(f"Predicted probability {away_team} wins: {1 - home_win_prob:.1%}")


if __name__ == "__main__":
    main()
