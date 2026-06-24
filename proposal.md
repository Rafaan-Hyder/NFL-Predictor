# Predicting NFL Game Outcomes Using Team Performance Statistics

**CPSC 483 — Machine Learning — Project Proposal**

## Group Members & Roles
Solo project. All roles — data collection, preprocessing, model development,
evaluation, and report writing — are handled by the single team member.

## Problem Description
The goal of this project is to predict the outcome (win or loss) of NFL games
based on team-level season statistics. NFL game prediction is a well-motivated
problem with applications in sports analytics, broadcasting, and strategic
planning. Despite the popularity of the topic, most publicly available
analyses rely on betting-market data or proprietary datasets. This project
focuses on freely available, official team performance statistics — such as
total yards, turnovers, points scored, third-down conversion rate, and sacks —
to build an interpretable classification model.

**Central question:** Can season-to-date team statistics reliably predict
which team wins a given game?

## Programming Language
Python 3, using:
- `pandas` / `NumPy` — data processing
- `scikit-learn` — model training and evaluation
- `matplotlib` / `seaborn` — visualization

## Dataset
Data sourced from **Pro Football Reference** (pro-football-reference.com),
which provides freely accessible historical NFL team statistics. Per-game and
season-aggregate stats will be collected for NFL seasons 2010–2023. Each
record represents one team's side of a matchup, with features derived from
season statistics *up to that point* (to avoid data leakage), labeled by
whether that team won or lost. Expected size: ~4,000–5,000 labeled samples.

## Approach
1. **Data Collection & Cleaning** — Download team game logs from Pro Football
   Reference. Clean missing values; engineer rolling-average features (yards
   per game, turnover differential, points per game) reflecting each team's
   recent form heading into a game.
2. **Preprocessing** — Build a matchup-level dataset pairing home/away team
   stats per game. Standard-scale continuous features. Encode home/away as a
   binary feature.
3. **Model Training** — Train and compare Logistic Regression (baseline),
   Decision Tree, Random Forest, and optionally a simple Neural Network, all
   via scikit-learn.
4. **Evaluation** — Stratified k-fold cross-validation. Report accuracy,
   precision, recall, F1, and ROC-AUC. Compare against the naive baseline
   (home team always wins, ~57% historically).
5. **Interpretation** — Use Random Forest feature importances to identify the
   strongest statistical predictors of game outcomes.

## Existing Code & Extensions
Scikit-learn provides the classifier implementations used; no project-specific
external code is reused. Original contributions:
1. Construction of a matchup-level NFL dataset from raw team game logs using
   rolling statistics to prevent data leakage.
2. Systematic comparison of multiple classifiers on this dataset.
3. Feature importance analysis identifying the strongest predictors of game
   outcomes.

**Data source:** Pro Football Reference (pro-football-reference.com)
**Language:** Python 3
**Libraries:** pandas, NumPy, scikit-learn, matplotlib
