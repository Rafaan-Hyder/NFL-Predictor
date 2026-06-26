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

Three files per season range are pulled:
  - games.csv              game-level schedule/results (one row per game,
                            final score for both teams) -> source of labels
  - stats_team_week_*.csv  team-game boxscore stats (one row per team per
                            week: passing/rushing yards, turnovers, sacks,
                            etc.) -> source of features
  - play_by_play_*.csv.gz  per-play data, used only to compute third-down
                            conversion rate and time of possession per team
                            per game (these two stats aren't in the boxscore
                            file). The full play-by-play is ~50k rows/season
                            and not needed downstream, so we aggregate it
                            down to one row per team-game immediately and
                            only keep that aggregate, not the raw pbp file.

All raw/aggregated outputs are saved to data/raw/. No matchup-level feature
engineering happens here (that's Phase 2) -- this script only collects and
lightly aggregates raw data.
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
PBP_URL_TMPL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.csv.gz"
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


def _time_str_to_seconds(t) -> float:
    """Convert an 'MM:SS' drive-time string to seconds; NaN-safe."""
    if pd.isna(t):
        return 0.0
    try:
        minutes, seconds = str(t).split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return 0.0


def fetch_situational_stats(out_dir: str) -> pd.DataFrame:
    """Download one season of play-by-play at a time, aggregate it down to
    per-team-game third-down conversion rate and time of possession, then
    discard the raw pbp file (it's ~50k rows/season and only these two
    aggregates are needed downstream)."""
    frames = []
    for season in range(START_SEASON, END_SEASON + 1):
        url = PBP_URL_TMPL.format(season=season)
        tmp_path = os.path.join(out_dir, f"_pbp_{season}.csv.gz")
        _download(url, tmp_path)
        pbp = pd.read_csv(tmp_path, compression="gzip", low_memory=False)
        os.remove(tmp_path)

        # Third-down rate: count of plays on 3rd down by offense (posteam)
        # that were converted vs. failed.
        third_downs = pbp[pbp["down"] == 3]
        third_down_pct = (
            third_downs.groupby(["game_id", "posteam"])
            .agg(
                third_down_conversions=("third_down_converted", "sum"),
                third_down_attempts=("third_down_converted", "count"),
            )
            .reset_index()
        )
        third_down_pct["third_down_pct"] = (
            third_down_pct["third_down_conversions"] / third_down_pct["third_down_attempts"]
        )

        # Time of possession: drive_time_of_possession is recorded per drive,
        # repeated across every play in that drive, so dedupe to one row per
        # drive before summing.
        drives = pbp.dropna(subset=["drive"]).drop_duplicates(subset=["game_id", "posteam", "drive"])
        drives = drives.copy()
        drives["top_seconds"] = drives["drive_time_of_possession"].apply(_time_str_to_seconds)
        top = drives.groupby(["game_id", "posteam"])["top_seconds"].sum().reset_index()

        season_stats = third_down_pct.merge(top, on=["game_id", "posteam"], how="outer")
        season_stats["season"] = season
        frames.append(season_stats)
        time.sleep(0.5)

    all_seasons = pd.concat(frames, ignore_index=True)
    all_seasons = all_seasons.rename(columns={"posteam": "team"})
    all_seasons.to_csv(os.path.join(out_dir, "team_game_situational.csv"), index=False)
    return all_seasons


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Fetching games.csv (schedule/results, {START_SEASON}-{END_SEASON})...")
    games = fetch_games(RAW_DIR)
    print(f"  -> {games.shape[0]} games, {games['season'].nunique()} seasons")

    print(f"Fetching team_week_stats.csv (boxscores, {START_SEASON}-{END_SEASON})...")
    team_week = fetch_team_week_stats(RAW_DIR)
    print(f"  -> {team_week.shape[0]} team-game rows, {team_week['season'].nunique()} seasons")

    print(f"\nFetching+aggregating play-by-play for third-down%/TOP ({START_SEASON}-{END_SEASON})...")
    situational = fetch_situational_stats(RAW_DIR)
    print(f"  -> {situational.shape[0]} team-game rows, {situational['season'].nunique()} seasons")

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
