"""Where everything lives. Every notebook and script imports its folders from here.

Paths are built from this file's own location, so they work no matter which
folder Python or Jupyter was started in.

aihack/
├── data/            competition files (read only)
├── notebooks/       01_eda ... 06_blend
├── scripts/         this file, features.py, tuning_loop.py
├── submissions/     files to upload
├── preds_for_blend/ out-of-fold and test predictions, for blending
├── tuning/          tuning trials and summary
├── docs/            brief and notes
└── logs/            CatBoost training logs

After the competition, delete data/, submissions/, preds_for_blend/, tuning/ and logs/,
plus notebook outputs, as the rules require.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
PREDS = ROOT / "preds_for_blend"
TUNING = ROOT / "tuning"
LOGS = ROOT / "logs"

for folder in (SUBMISSIONS, PREDS, TUNING, LOGS):
    folder.mkdir(exist_ok=True)
