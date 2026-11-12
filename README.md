# AIFUL 12-month default prediction

Predicts the probability that a loan account is in default 12 months after origination, from application data. Submissions are scored by ROC-AUC.

## Layout

| Folder | Contents | In git |
|---|---|---|
| `data/` | Competition files: train, test, submission template, data dictionary | No |
| `notebooks/` | `01_eda` to `06_blend`, described below | Yes |
| `scripts/` | `paths.py` (folder locations), `features.py` (shared features), `tuning_loop.py` (hyperparameter tuning) | Yes |
| `docs/` | Competition brief and notes | Yes |
| `submissions/` | Files to upload | No |
| `preds_for_blend/` | Out-of-fold and test predictions that `06_blend` combines | No |
| `tuning/` | Tuning trials and summary from `tuning_loop.py` | No |
| `logs/` | CatBoost training logs | No |

Every notebook and script gets its folder locations from `scripts/paths.py`, so they work wherever Jupyter or Python is started.

## Setup

1. Put the 4 competition files in `data/`.
2. Create an environment with Python 3.12 and install the packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running

| Step | File | What it does |
|---:|---|---|
| 1 | `notebooks/01_eda.ipynb` | Explores the data |
| 2 | `notebooks/02_baseline.ipynb` | Sets up cross-validation |
| 3 | `notebooks/03_lightgbm.ipynb` | Trains LightGBM, writes a submission and predictions for blending, and tests feature ideas |
| 4 | `notebooks/04_catboost.ipynb` | Trains CatBoost, writes a submission and predictions for blending |
| 5 | `notebooks/05_logreg.ipynb` | Trains logistic regression, writes a submission and predictions for blending |
| 6 | `notebooks/06_blend.ipynb` | Rank-averages the saved predictions and writes the blended submission |

Steps 3 to 5 can run in any order, but all of them must finish before step 6.

To tune LightGBM and CatBoost, run this from the project folder. It takes about an hour:

```bash
python scripts/tuning_loop.py
```

To check that the tuning code runs, with 2 trials per model:

```bash
QUICK=1 python scripts/tuning_loop.py
```

## Validation

All models use the same 5-fold stratified split, with seed 42, so their out-of-fold predictions can be compared and blended. Feature and tuning decisions are also rechecked on a second split, with seed 7. The results log is at the top of `scripts/features.py`.

## Data rules

Competition data must not leave this machine and must be deleted after the competition, including copies. This repository tracks code and notes only:

- `.gitignore` excludes the data and everything generated from it.
- **Clear notebook outputs before committing.** Saved outputs contain statistics computed from the competition data.
