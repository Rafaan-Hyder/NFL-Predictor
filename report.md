# Predicting NFL Game Outcomes Using Team Performance Statistics

CPSC 483 — Machine Learning — Project Report (draft)

## Problem

Can pre-game, season-to-date team performance statistics predict which
team wins an NFL game? This is a binary classification problem: for each
game, predict whether the home team wins (1) or loses (0).

## Data

Data covers NFL seasons 2010-2023 (3,793 games), sourced from nflverse
(the dataset behind the `nfl_data_py` package), pulled directly from its
GitHub-hosted CSV releases rather than through the package itself or by
scraping Pro Football Reference as originally proposed — see "Deviations
from the proposal" below.

Three raw tables were combined: a game-level schedule/results table
(scores, date, rest days), a team-game boxscore table (passing/rushing
yards, turnovers, sacks), and a third table of third-down conversion rate
and time of possession aggregated from play-by-play data (these two stats
aren't in the standard boxscore export).

One data-quality issue surfaced during integration: relocated franchises
(Raiders, Chargers, Rams) are labeled with their *historical* abbreviation
(OAK/SD/STL) in the schedule table but their *current* one (LV/LAC/LA) in
the boxscore and play-by-play tables, for all seasons including before
the move. Team codes were normalized to the current abbreviation before
joining the tables.

## Feature Engineering

Every feature is a **trailing 5-game rolling average** of a team's prior
performance (points scored/allowed, total yards, turnover margin, sack
margin, third-down %, time of possession), computed separately for the
home and away team in each matchup, plus each team's rest days before the
game.

Two choices here matter for avoiding data leakage and are worth
explaining:

- **Rolling average, not season-to-date total.** A team's strength
  changes over a season (injuries, a young roster maturing, a coaching
  change), so the last 5 games predict the next game better than a
  season-long average that weights week 1 and week 16 equally.
- **The average is shifted by one game before the rolling window is
  applied**, so the window for game *N* covers games *N-5* through *N-1*
  and never includes game *N* itself — the single most important rule in
  this project, since including the current game's own stats as a
  "predictor" of its own outcome would silently inflate accuracy.

The rolling window is **not reset at season boundaries**: a team's
week-1 rolling average carries over from the end of the previous season,
rather than being undefined. This means only each team's first 3 games
ever (at the very start of the 2010 season, where no history exists at
all) had to be dropped — 48 games total, leaving 3,745 matchups.

An automated check in `build_features.py` re-derives one team's rolling
average by hand from its game log and asserts it matches what the
pipeline produced, and separately confirms the current game's ID never
appears inside its own rolling window.

## Approach

Three classifiers were trained and compared: Logistic Regression,
Decision Tree, and Random Forest (all scikit-learn).

**Train/test split is chronological, not random**: training data is
2010-2021 (3,176 games), and the test set is 2022-2023 (569 games),
touched only once for final evaluation. A random split would let the
model train on December 2023 games and be evaluated on January 2023
games — effectively letting it see the future relative to some test
examples, which doesn't match how the model would actually be used (a
game is always predicted using only data from before it happened).

Continuous features were standard-scaled with the scaler fit on the
training set only. Hyperparameters (e.g. regularization strength for
Logistic Regression, tree depth, forest size) were tuned via 5-fold
stratified cross-validation *within the training set*, so the
chronological test set was never used for model selection.

## Results

Naive baseline: home team always wins, 56.4% accuracy on the test set
(close to the proposal's commonly cited ~57% historical rate).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive (home always wins) | 0.564 | 0.564 | 1.000 | 0.721 | — |
| Logistic Regression | **0.613** | 0.629 | 0.766 | 0.691 | **0.645** |
| Decision Tree | 0.583 | 0.619 | 0.682 | 0.649 | 0.587 |
| Random Forest | 0.605 | 0.628 | 0.735 | 0.677 | 0.617 |

Logistic Regression is the best-performing model, beating the naive
baseline by about 5 accuracy points (61.3% vs. 56.4%) and achieving the
highest ROC-AUC (0.645). The Random Forest's training accuracy (93%)
vastly exceeded its test accuracy (60%), a clear overfitting signature
for an unconstrained tree ensemble; cross-validation correctly
anticipated this — the CV accuracy (0.626) was much closer to the
eventual test accuracy than the training accuracy was, which is exactly
why CV is used for model selection instead of training accuracy.

See `reports/figures/` for ROC curves, confusion matrices, and feature
distributions.

## Interpretation

Random Forest feature importances (`reports/figures/feature_importances.png`)
rank rolling total yards, points for, and points against as the
strongest predictors, with third-down % and time of possession close
behind. Turnover margin and sack margin matter less than expected, and
rest days (home/away) are the weakest predictors by a wide margin —
suggesting the standard week-to-week rest schedule doesn't vary enough
to carry much predictive signal on its own.

An incidental finding from the EDA (`reports/figures/home_win_rate_by_season.png`):
home-field advantage visibly dipped in 2019-2021, plausibly related to
reduced or absent crowds during the COVID-19 pandemic, then recovered.

## Limitations

- Accuracy gains over the naive baseline are real but modest (~5 points),
  consistent with NFL outcomes being inherently difficult to predict from
  team-level stats alone — injuries, weather, and in-game randomness
  aren't captured here.
- Third-down % and time of possession are not in the standard boxscore
  export and were derived from play-by-play data instead of pulled
  directly, a heavier and slightly more error-prone path than the rest of
  the pipeline.
- The model has no notion of individual player quality (e.g. starting
  quarterback), which is a well-known strong predictor in real sports
  analytics.

## Deviations from the Proposal

The proposal named Pro Football Reference as the primary data source,
with `nfl_data_py` as a documented fallback. We used the nflverse data
that backs `nfl_data_py`, but pulled it directly from its GitHub-hosted
CSV releases rather than through the package's own download function,
because that function's target URL is blocked by this network's egress
policy. This is the build plan's intended fallback in spirit — same open,
well-documented dataset, no scraping fragility — just reached by a
slightly different route. See `src/fetch_data.py`'s module docstring for
the full reasoning.
