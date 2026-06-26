"""
Phase 2 — Feature Engineering.

Data-leakage rule for this whole script: every feature describing a team
going into a game must be computed strictly from games *before* that game.
We enforce this mechanically, not just by convention: rolling stats are
built with `.shift(1)` before `.rolling(...)`, so the window for game N
covers games [N-5, N-1] and never N itself. There's an explicit assertion
at the bottom of main() that re-derives one game's features by hand and
checks they match, as a second line of defense against a future edit
silently breaking this.

Rolling window, not season totals: we use a trailing 5-game rolling average
(not season-to-date totals) because team strength changes over a season
(injuries, trades, a young team improving) -- a team's last 5 games predict
their next game much better than per-game averages that weight September
and December equally. The window is also not reset at season boundaries:
a team's week-1 rolling average is computed from the end of last season
when at least one prior game exists, which is the build plan's documented
fallback for early-season rows (rather than dropping every team's first
2-3 games of every season, only each team's first few games of 2010 -- the
very start of our data window, where no prior history exists at all -- get
dropped).
"""

import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ROLLING_WINDOW = 5
MIN_PRIOR_GAMES = 3  # need at least this many past games before we trust the rolling average

# Per-team-game features computed from raw boxscore + situational stats,
# before any rolling is applied. These are the *inputs* to the rolling
# averages below, not features themselves (a single game's total yards
# is just as much "this game's outcome" as the score is).
RAW_FEATURE_COLS = [
    "points_for",
    "points_against",
    "total_yards",
    "turnover_margin",
    "sack_margin",
    "third_down_pct",
    "top_seconds",
]


# games.csv labels teams by the abbreviation used at the time (e.g. the
# Raiders are "OAK" through 2019), while team_week_stats.csv and the
# play-by-play aggregates always use each franchise's current abbreviation.
# Relabel games.csv to the current codes so every source joins on the same
# team key.
TEAM_CODE_RENAME = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def load_raw():
    games = pd.read_csv(os.path.join(RAW_DIR, "games.csv"))
    team_week = pd.read_csv(os.path.join(RAW_DIR, "team_week_stats.csv"))
    situational = pd.read_csv(os.path.join(RAW_DIR, "team_game_situational.csv"))

    games["home_team"] = games["home_team"].replace(TEAM_CODE_RENAME)
    games["away_team"] = games["away_team"].replace(TEAM_CODE_RENAME)

    return games, team_week, situational


def build_team_game_log(games: pd.DataFrame, team_week: pd.DataFrame, situational: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one-row-per-game to one-row-per-team-per-game (a 'team
    game log'), with derived stats. This long format is what we compute
    rolling averages over, before pivoting back to matchup rows."""

    # Derived per-game stats. "Turnovers" and "sacks" each have an offense
    # and defense side, so we track the net margin for both -- a team that
    # forces more turnovers/sacks than it commits is what actually predicts
    # winning, not the raw counts in isolation.
    team_week["total_yards"] = team_week["passing_yards"] + team_week["rushing_yards"]
    turnovers_lost = (
        team_week["passing_interceptions"]
        + team_week["sack_fumbles_lost"]
        + team_week["rushing_fumbles_lost"]
        + team_week["receiving_fumbles_lost"]
    )
    turnovers_forced = team_week["def_interceptions"] + team_week["fumble_recovery_opp"]
    team_week["turnover_margin"] = turnovers_forced - turnovers_lost
    team_week["sack_margin"] = team_week["def_sacks"] - team_week["sacks_suffered"]

    # Attach game-level context (date, home/away, opponent score) by joining
    # each team-game row to its game and figuring out which side it was on.
    home = games[["game_id", "gameday", "season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"]].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["is_home"] = 1

    away = games[["game_id", "gameday", "season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"]].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["is_home"] = 0

    context_cols = ["game_id", "gameday", "season", "week", "game_type", "team", "opponent", "points_for", "points_against", "is_home"]
    game_context = pd.concat([home[context_cols], away[context_cols]], ignore_index=True)

    # situational's game_id is reliable for every season (it's built from
    # play-by-play, which always has it), so that merge uses game_id+team.
    log = game_context.merge(
        situational[["game_id", "team", "third_down_pct", "top_seconds"]],
        on=["game_id", "team"],
        how="left",
    )

    # team_week_stats.csv's own game_id column is only populated from 2022
    # onward, so we join it on season+week+team instead -- a reliable key
    # since a team plays at most one game per week.
    log = log.merge(
        team_week[["season", "week", "team", "total_yards", "turnover_margin", "sack_margin"]],
        on=["season", "week", "team"],
        how="left",
    )

    log["gameday"] = pd.to_datetime(log["gameday"])
    log = log.sort_values(["team", "gameday"]).reset_index(drop=True)
    return log


def add_rolling_features(log: pd.DataFrame) -> pd.DataFrame:
    """Add trailing rolling-average columns for each raw feature, shifted by
    one game so the window for game N never includes game N itself."""
    log = log.sort_values(["team", "gameday"]).reset_index(drop=True)
    grouped = log.groupby("team", group_keys=False)

    for col in RAW_FEATURE_COLS:
        shifted = grouped[col].shift(1)
        log[f"roll_{col}"] = shifted.groupby(log["team"]).transform(
            lambda s: s.rolling(ROLLING_WINDOW, min_periods=MIN_PRIOR_GAMES).mean()
        )

    # How many prior games this team has played (used to drop rows with
    # insufficient history below).
    log["games_played_before"] = grouped.cumcount()
    return log


def build_matchups(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Pivot the team game log (one row per team-game, with rolling features
    already attached) into one row per game, with home_*/away_* columns."""
    roll_cols = [f"roll_{c}" for c in RAW_FEATURE_COLS]

    home_side = log[log["is_home"] == 1][["game_id", "games_played_before"] + roll_cols]
    home_side = home_side.rename(columns={c: f"home_{c}" for c in roll_cols})
    home_side = home_side.rename(columns={"games_played_before": "home_games_played_before"})

    away_side = log[log["is_home"] == 0][["game_id", "games_played_before"] + roll_cols]
    away_side = away_side.rename(columns={c: f"away_{c}" for c in roll_cols})
    away_side = away_side.rename(columns={"games_played_before": "away_games_played_before"})

    matchups = games[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_rest", "away_rest", "result"]].copy()
    matchups = matchups.merge(home_side, on="game_id", how="left")
    matchups = matchups.merge(away_side, on="game_id", how="left")

    # Label: did the home team win? Ties (result == 0) are extremely rare
    # in the NFL (~0.1% of games) and aren't a "win" for either side, so we
    # label them 0 along with losses rather than dropping them.
    matchups["home_win"] = (matchups["result"] > 0).astype(int)

    # Drop games where either team doesn't yet have enough rolling history
    # (only affects each team's first few games at the very start of 2010,
    # since the rolling window otherwise carries over across season
    # boundaries -- see module docstring).
    has_history = (
        (matchups["home_games_played_before"] >= MIN_PRIOR_GAMES)
        & (matchups["away_games_played_before"] >= MIN_PRIOR_GAMES)
    )
    dropped = (~has_history).sum()
    matchups = matchups[has_history].drop(columns=["home_games_played_before", "away_games_played_before"])

    print(f"Dropped {dropped} games with insufficient rolling history for one or both teams.")
    return matchups


def verify_no_leakage(log: pd.DataFrame, team: str = None):
    """Recompute one team's rolling features by hand from their game log and
    check it matches what add_rolling_features produced, confirming the
    window excludes the current game."""
    if team is None:
        team = log["team"].iloc[0]
    team_log = log[log["team"] == team].sort_values("gameday").reset_index(drop=True)

    # Pick a game far enough into the log to have a full window.
    idx = MIN_PRIOR_GAMES + 2
    if idx >= len(team_log):
        return  # not enough games for this team to check; harmless

    window = team_log.loc[idx - ROLLING_WINDOW : idx - 1, "points_for"]
    expected = window.mean()
    actual = team_log.loc[idx, "roll_points_for"]
    assert abs(expected - actual) < 1e-9, (
        f"Leakage check failed for {team} at index {idx}: "
        f"expected {expected}, got {actual}"
    )

    # Also confirm the window strictly precedes the game in question.
    current_game_id = team_log.loc[idx, "game_id"]
    window_game_ids = team_log.loc[idx - ROLLING_WINDOW : idx - 1, "game_id"]
    assert current_game_id not in window_game_ids.values, (
        f"Leakage check failed: game {current_game_id} appears in its own rolling window"
    )

    print(f"Leakage check passed for team={team}, game={current_game_id}: "
          f"roll_points_for matches hand-computed average of the 5 prior games, "
          f"and the current game is not among them.")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    games, team_week, situational = load_raw()
    log = build_team_game_log(games, team_week, situational)
    log = add_rolling_features(log)

    verify_no_leakage(log)

    matchups = build_matchups(log, games)

    nan_counts = matchups.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print("\nColumns with remaining NaNs:")
        print(nan_counts)
    else:
        print("\nNo NaNs remaining in matchups.csv.")

    out_path = os.path.join(PROCESSED_DIR, "matchups.csv")
    matchups.to_csv(out_path, index=False)
    print(f"\nSaved {matchups.shape[0]} rows x {matchups.shape[1]} cols to {out_path}")
    print(f"Home win rate: {matchups['home_win'].mean():.3f}")


if __name__ == "__main__":
    main()
