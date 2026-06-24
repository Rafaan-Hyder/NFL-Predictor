"""
Phase 1 — Data Acquisition.

Data source decision:
The proposal named Pro Football Reference as the primary source, with
nfl_data_py (nflverse data) as the documented fallback. We went straight to
the fallback: nfl_data_py's own download helper (import_schedules) hits
http://www.habitatring.com, which this network's egress policy blocks
(HTTP 403). The same underlying nflverse data is published as versioned CSV
releases on GitHub (raw.githubusercontent.com / github.com/nflverse), which
*is* reachable, so we download directly from there instead of going through
the nfl_data_py package. This avoids scraping fragility entirely while still
using the same well-documented, open dataset the build plan recommended.

Two files per season range are pulled:
  - games.csv            game-level schedule/results (one row per game,
                          final score for both teams) -> source of labels
  - stats_team_week_*.csv  team-game boxscore stats (one row per team per
                          week: passing/rushing yards, turnovers, sacks,
                          etc.) -> source of features

Both are saved untouched to data/raw/. No feature engineering happens here
(that's Phase 2) -- this script only collects raw data.
"""

import os
import subprocess
import time

import pandas as pd

START_SEASON = 2010
END_SEASON = 2023
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_WEEK_URL_TMPL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)


def _download(url: str, dest_path: str, retries: int = 3) -> None:
    """Download a URL to disk via curl; both pandas' and urllib's HTTP readers
    truncated larger files when read through this network's proxy, but curl
    (with -L to follow GitHub's release redirects) fetches them intact."""
    last_err = None
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-sSL", "-o", dest_path, url], capture_output=True, text=True
        )
        if result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return
        last_err = RuntimeError(f"curl failed for {url}: {result.stderr}")
        time.sleep(2 ** attempt)
    raise last_err


def fetch_games(out_dir: str) -> pd.DataFrame:
    """Download the full game-level schedule/results table and trim to our seasons."""
    dest = os.path.join(out_dir, "games.csv")
    _download(GAMES_URL, dest)
    df = pd.read_csv(dest, low_memory=False)
    df = df[(df["season"] >= START_SEASON) & (df["season"] <= END_SEASON)]
    df.to_csv(dest, index=False)
    return df


def fetch_team_week_stats(out_dir: str) -> pd.DataFrame:
    """Download one team-game boxscore file per season and concatenate."""
    frames = []
    for season in range(START_SEASON, END_SEASON + 1):
        url = TEAM_WEEK_URL_TMPL.format(season=season)
        tmp_path = os.path.join(out_dir, f"_team_week_{season}.csv")
        _download(url, tmp_path)
        frames.append(pd.read_csv(tmp_path, low_memory=False))
        os.remove(tmp_path)
        time.sleep(0.5)  # be polite even though these are static release assets
    all_seasons = pd.concat(frames, ignore_index=True)
    all_seasons.to_csv(os.path.join(out_dir, "team_week_stats.csv"), index=False)
    return all_seasons


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Fetching games.csv (schedule/results, {START_SEASON}-{END_SEASON})...")
    games = fetch_games(RAW_DIR)
    print(f"  -> {games.shape[0]} games, {games['season'].nunique()} seasons")

    print(f"Fetching team_week_stats.csv (boxscores, {START_SEASON}-{END_SEASON})...")
    team_week = fetch_team_week_stats(RAW_DIR)
    print(f"  -> {team_week.shape[0]} team-game rows, {team_week['season'].nunique()} seasons")

    # Sanity check: regular season is 256 games/season post-2002 (272 from 2021
    # onward after the 17th week was added), so 2 team-rows per game roughly
    # matches team_week row counts once playoffs are excluded.
    reg_games_per_season = games[games["game_type"] == "REG"].groupby("season").size()
    print("\nRegular-season games per season:")
    print(reg_games_per_season)

    missing_seasons = set(range(START_SEASON, END_SEASON + 1)) - set(games["season"].unique())
    if missing_seasons:
        print(f"WARNING: missing seasons in games.csv: {sorted(missing_seasons)}")
    else:
        print("\nNo missing seasons in games.csv.")


if __name__ == "__main__":
    main()
