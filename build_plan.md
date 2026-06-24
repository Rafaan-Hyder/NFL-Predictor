Build Plan — NFL Game Outcome Predictor

This plan breaks the project (see proposal.md) into phases sized for a
one-week, solo timeline. Each phase has a clear deliverable and a "done when"
check so progress is verifiable at every step. Follow them in order — don't
start model training before the data pipeline is validated.


Phase 0 — Project Setup

Goal: A clean, runnable repo skeleton.


 requirements.txt (pandas, numpy, scikit-learn, matplotlib, seaborn,
requests, beautifulsoup4 or nfl_data_py, jupyter optional)
 Folder structure:


  data/raw/          # untouched downloaded data
  data/processed/    # cleaned, feature-engineered data
  src/               # all source code, no notebooks
  notebooks/         # optional exploration only, not the deliverable
  reports/figures/   # saved plots for the final report


 .gitignore for data/raw (if large) and venv/cache files
 Virtual environment created and dependencies installed


Done when: python -c "import pandas, sklearn, matplotlib" runs clean.


Phase 1 — Data Acquisition

Goal: Raw per-game team stats for NFL seasons 2010–2023 saved locally.


 Decide data source:

Preferred: nfl_data_py (Python package wrapping nflverse/nflfastR data,
open and well-documented, avoids scraping fragility)
Fallback: scrape Pro Football Reference team game logs with
requests + BeautifulSoup, with realistic rate-limiting
Pick one and document the choice + reasoning at the top of the script



 Script: src/fetch_data.py — downloads/builds raw season-by-season
team game logs, saves to data/raw/
 Sanity check: confirm row counts roughly match expected NFL game counts
per season (256 regular-season games × 2 team-rows per game, etc.)


Done when: data/raw/ contains all seasons 2010–2023 (or the adjusted
range) with no missing seasons, and a quick .head() / .info() printout
looks sane.


Phase 2 — Feature Engineering (data leakage is the main risk here)

Goal: A matchup-level dataset where every feature is knowable before
kickoff.


 Script: src/build_features.py
 For each team, compute rolling averages (e.g., trailing 3–5 games and/or
season-to-date) for: points scored/allowed, total yards, passing yards,
rushing yards, turnovers, third-down conversion %, sacks, time of
possession
 Merge into matchup rows: one row per game with home-team rolling stats,
away-team rolling stats, home/away flag, and the label (home team
win = 1 / loss = 0)
 Explicitly verify no feature uses same-game outcome data (this is the
#1 way these projects silently get inflated accuracy)
 Handle early-season games with insufficient rolling history (e.g. drop
first 2–3 games per season, or fall back to prior-season averages)
 Save to data/processed/matchups.csv


Done when: data/processed/matchups.csv has one row per game, a binary
label column, no NaNs (or a documented justified handling), and a written
note confirming no leakage.


Phase 3 — Exploratory Data Analysis

Goal: Understand the data before modeling; produces a couple of figures
for the report.


 Class balance check (home win rate — should land near the ~57%
historical baseline)
 Feature distributions, correlation heatmap
 2–3 saved figures to reports/figures/


Done when: You can state the naive baseline accuracy in one sentence and
have at least one correlation/distribution figure saved.


Phase 4 — Modeling

Goal: Multiple trained classifiers, fairly compared.


 Script: src/train_models.py
 Train/test split — must be chronological, not random (e.g. train
on 2010–2021, test on 2022–2023) to simulate real predictive use and
avoid leaking future info into the past
 Standard-scale continuous features (fit scaler on train only)
 Models: Logistic Regression (baseline), Decision Tree, Random Forest;
Neural Network (sklearn MLPClassifier) optional stretch goal
 Use stratified k-fold cross-validation on the training set for model
selection/tuning; reserve chronological test set for final evaluation
 Save trained models (joblib) to models/


Done when: All models train without errors and produce predictions on
the held-out test set.


Phase 5 — Evaluation

Goal: Objective, comparable performance numbers.


 Script: src/evaluate.py
 Metrics per model: accuracy, precision, recall, F1, ROC-AUC
 Comparison table: all models + naive home-team-always-wins baseline
 ROC curve plot, confusion matrix per model → save to reports/figures/
 Random Forest feature importance plot (ties back to "Interpretation"
step in the proposal)


Done when: You have a results table and at least 3 saved figures, and
you can say in one sentence whether the model beats the naive baseline.


Phase 6 — Report & Polish

Goal: Deliverables matching the grading rubric (written report, clean
code, presentation-ready figures).


 README.md — how to run the project end-to-end, summary of results
 Written report draft covering: problem, data, approach, results,
discussion of which stats were most predictive, limitations
 Clean up src/ scripts (docstrings, remove dead code)
 Optional: simple script/notebook to predict a single hypothetical
matchup, useful for a live demo during the presentation


Done when: A new person could clone the repo, run one or two commands,
and reproduce your main result.


Stretch Goals (only if time allows after Phase 6)


Add Vegas point-spread comparison (does the model add value over the
betting line?) — be careful, this is a step up in data sourcing
Try gradient boosting (XGBoost / LightGBM) for a stronger model
Add a small Streamlit demo app for the presentation
