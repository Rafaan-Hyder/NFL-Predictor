# NFL Game Outcome Predictor

CPSC 483 course project. Predicts NFL game winners from each team's
rolling pre-game form (points, yards, turnovers, sacks, third-down %,
time of possession) using Logistic Regression, a Decision Tree, and a
Random Forest. See `proposal.md` for the original project proposal and
`build_plan.md` for the phase-by-phase build plan this was implemented
against.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the full pipeline

Run in order — each script reads the previous step's output from
`data/raw/` or `data/processed/`:

```bash
python src/fetch_data.py        # Phase 1: download raw data to data/raw/
python src/build_features.py    # Phase 2: rolling features -> data/processed/matchups.csv
python src/eda.py               # Phase 3: figures -> reports/figures/
python src/train_models.py      # Phase 4: train + tune models -> models/
python src/evaluate.py          # Phase 5: metrics + figures -> reports/figures/, data/processed/results_table.csv
```

`data/processed/matchups.csv` and the trained models in `models/` are
already committed, so later steps can be re-run without re-downloading
or re-training anything earlier in the pipeline.

## Live demo

Predict a hypothetical matchup using each team's most recent rolling
form in the dataset:

```bash
python src/predict_matchup.py KC SF
```

## Results summary

Trained on 2010-2021, evaluated on a held-out chronological test set
(2022-2023 seasons, 569 games):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive (home team always wins) | 0.564 | 0.564 | 1.000 | 0.721 | — |
| Logistic Regression | **0.613** | 0.629 | 0.766 | 0.691 | **0.645** |
| Decision Tree | 0.583 | 0.619 | 0.682 | 0.649 | 0.587 |
| Random Forest | 0.605 | 0.628 | 0.735 | 0.677 | 0.617 |

Logistic Regression is the best-performing model and beats the naive
home-field baseline by about 5 points of accuracy. See `reports/figures/`
for ROC curves, confusion matrices, and Random Forest feature
importances, and `report.md` for the full write-up.

## Project structure

```
data/raw/          downloaded data (games, boxscores, situational stats)
data/processed/    matchups.csv (final feature set) and train/test splits
src/               all source code (see below)
notebooks/         scratch exploration only, not a deliverable
reports/figures/   saved plots
models/            trained models (joblib) + fitted scaler
```

| Script | Phase | Purpose |
|---|---|---|
| `src/fetch_data.py` | 1 | Download raw game/boxscore/play-by-play data |
| `src/build_features.py` | 2 | Build leakage-safe rolling features, one row per game |
| `src/eda.py` | 3 | Class balance, distributions, correlation heatmap |
| `src/train_models.py` | 4 | Chronological split, CV tuning, train models |
| `src/evaluate.py` | 5 | Metrics table, ROC/confusion matrix/importance plots |
| `src/predict_matchup.py` | 6 | Demo: predict one hypothetical matchup |

## Data source note

The proposal named Pro Football Reference as the primary data source.
We used nflverse data (the same data underlying the `nfl_data_py`
package) instead, pulled directly from its GitHub-hosted CSV releases.
This was a documented fallback from the build plan, chosen because it's
the same open, well-documented dataset without scraping fragility. See
the module docstring in `src/fetch_data.py` for the full reasoning.
