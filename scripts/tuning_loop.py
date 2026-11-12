"""Tune LightGBM and CatBoost, then blend them.

Run from the project folder:
       python scripts/tuning_loop.py           full run, about an hour
       QUICK=1 python scripts/tuning_loop.py   2 trials each, only checks the code runs
Outputs go to tuning/ and preds_for_blend/ (quick runs: tuning_quick/), wherever you run it from.

Steps
1. Score the current notebook settings (the baseline) on the usual 5 folds, seed 42.
2. Tune each model with Optuna on those same folds, with early stopping on each
   validation fold. The search uses a faster learning rate to save time.
3. Re-score the best settings at the notebooks' slower learning rate and keep
   whichever scores higher.
4. Find the rank-blend weight that gives the best mean fold AUC.
5. Re-check tuned against current settings on a different fold split (seed 7).
   A gain that only exists on one split is probably luck.
6. Train final models on all rows and save out-of-fold and test predictions.

Only summary numbers are printed, never customer rows.
"""

import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import add_features
from paths import DATA as DATA_DIR, PREDS, ROOT, TUNING

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

QUICK = os.environ.get("QUICK") == "1"
TARGET = "12-Month Default Flag"
SEED, CHECK_SEED, N_SPLITS = 42, 7, 5
OUT = ROOT / "tuning_quick" if QUICK else TUNING
OUT.mkdir(exist_ok=True)
PRED_OUT = OUT if QUICK else PREDS
PRED_OUT.mkdir(exist_ok=True)

N_TRIALS = {"lgbm": 2 if QUICK else 80, "catboost": 2 if QUICK else 40}
TIMEOUT = {"lgbm": None if QUICK else 25 * 60, "catboost": None if QUICK else 50 * 60}
SEARCH_LR = {"lgbm": 0.1 if QUICK else 0.03, "catboost": 0.15 if QUICK else 0.06}
FINAL_LR = {"lgbm": 0.1 if QUICK else 0.01, "catboost": 0.15 if QUICK else 0.03}
PATIENCE = 50 if QUICK else 200

CATS = [
    "Application Channel", "Occupation Category", "Industry Category",
    "Employment Type Category", "Insurance Type", "Company Size Category",
    "Business Entity Type", "Residence Type", "Residence Ownership Status",
]

# The settings currently in 03_lightgbm.ipynb and 04_catboost.ipynb (plus library
# defaults for anything they don't set). Used as the baseline and as trial 0.
CURRENT = {
    "lgbm": dict(num_leaves=8, min_child_samples=100, subsample=0.8,
                 colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1e-3,
                 cat_smooth=10.0, min_data_per_group=100),
    "catboost": dict(depth=6, l2_leaf_reg=5.0, random_strength=1.0,
                     subsample=0.8, rsm=1.0, one_hot_max_size=2),
}


def log(msg):
    print(msg, flush=True)


# ---------- data ----------
train = pd.read_csv(DATA_DIR / "train_dataset.csv")
test = pd.read_csv(DATA_DIR / "test_without_gt.csv")
X = add_features(train.drop(columns=["ID", TARGET]))
y = train[TARGET]
X_test = add_features(test)[X.columns]

X_lgb, X_test_lgb = X.copy(), X_test.copy()
for c in CATS:
    X_lgb[c] = X_lgb[c].astype("category")
    known = X_lgb[c].cat.categories
    X_test_lgb[c] = pd.Categorical(X_test_lgb[c].where(X_test_lgb[c].isin(known)), categories=known)

X_cb, X_test_cb = X.copy(), X_test.copy()
for c in CATS:
    X_cb[c] = X_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

DATA = {"lgbm": (X_lgb, X_test_lgb), "catboost": (X_cb, X_test_cb)}
log(f"rows {len(y)}, features {X.shape[1]}, test rows {len(X_test)}, "
    f"unseen test category values {int(X_test_lgb[CATS].isna().sum().sum())}")


# ---------- one model, one fold ----------
def make(name, params, lr, n_estimators=5000):
    if name == "lgbm":
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=lr, subsample_freq=1,
                                  random_state=SEED, verbose=-1, **params)
    return CatBoostClassifier(iterations=n_estimators, learning_rate=lr, eval_metric="AUC",
                              cat_features=CATS, random_seed=SEED, thread_count=-1,
                              verbose=0, allow_writing_files=False, **params)


def fit(name, model, X_fit, y_fit, X_val, y_val):
    """Fit with early stopping and return the number of trees kept."""
    if name == "lgbm":
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        return model.best_iteration_
    model.fit(X_fit, y_fit, eval_set=(X_val, y_val), early_stopping_rounds=PATIENCE)
    return model.get_best_iteration() + 1


def run_cv(name, params, lr, seed=SEED, trial=None):
    X_tr = DATA[name][0]
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    scores, trees = [], []
    for k, (fit_idx, val_idx) in enumerate(folds.split(X_tr, y)):
        model = make(name, params, lr)
        trees.append(fit(name, model, X_tr.iloc[fit_idx], y.iloc[fit_idx],
                         X_tr.iloc[val_idx], y.iloc[val_idx]))
        oof[val_idx] = model.predict_proba(X_tr.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], oof[val_idx]))
        if trial is not None:
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return np.array(scores), oof, trees


# ---------- search spaces ----------
def suggest(name, t):
    if name == "lgbm":
        return dict(
            num_leaves=t.suggest_int("num_leaves", 4, 64, log=True),
            min_child_samples=t.suggest_int("min_child_samples", 20, 1000, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 50, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
            cat_smooth=t.suggest_float("cat_smooth", 1, 100, log=True),
            min_data_per_group=t.suggest_int("min_data_per_group", 10, 500, log=True),
        )
    return dict(
        depth=t.suggest_int("depth", 4, 8),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 50, log=True),
        random_strength=t.suggest_float("random_strength", 0.1, 20, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        rsm=t.suggest_float("rsm", 0.5, 1.0),
        one_hot_max_size=t.suggest_categorical("one_hot_max_size", [2, 4, 10, 20]),
    )


def tune(name):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.enqueue_trial(CURRENT[name])

    def objective(t):
        scores, _, trees = run_cv(name, suggest(name, t), SEARCH_LR[name], trial=t)
        t.set_user_attr("folds", scores.round(4).tolist())
        t.set_user_attr("trees", int(np.mean(trees)))
        return float(scores.mean())

    def progress(st, t):
        result = "pruned" if t.state == optuna.trial.TrialState.PRUNED else f"{t.value:.4f}"
        log(f"[{name}] trial {t.number:3d}: {result:>7}   best so far {st.best_value:.4f}")

    study.optimize(objective, n_trials=N_TRIALS[name], timeout=TIMEOUT[name], callbacks=[progress])
    study.trials_dataframe().to_csv(OUT / f"{name}_trials.csv", index=False)
    return study


def fmt(scores):
    return f"{scores.mean():.4f}  folds {np.round(scores, 4).tolist()}"


summary = {"quick": QUICK}
results = {}
t0 = time.time()

for name in ["lgbm", "catboost"]:
    log(f"\n===== {name} =====")
    base_scores, base_oof, base_trees = run_cv(name, CURRENT[name], FINAL_LR[name])
    log(f"baseline (current settings, lr {FINAL_LR[name]}): {fmt(base_scores)}")

    study = tune(name)
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    log(f"search finished: {n_done} complete, {n_pruned} pruned, best {study.best_value:.4f} "
        f"at lr {SEARCH_LR[name]}")

    best = study.best_params
    candidates = {}
    for lr in sorted({SEARCH_LR[name], FINAL_LR[name]}):
        s, o, tr = run_cv(name, best, lr)
        candidates[lr] = (s, o, tr)
        log(f"best settings at lr {lr}: {fmt(s)}  (mean trees {int(np.mean(tr))})")
    lr_pick = max(candidates, key=lambda lr: candidates[lr][0].mean())
    s, o, tr = candidates[lr_pick]

    results[name] = dict(params=best, lr=lr_pick, scores=s, oof=o, trees=tr,
                         base_scores=base_scores, base_oof=base_oof)
    summary[name] = dict(
        baseline_auc=round(float(base_scores.mean()), 4),
        tuned_auc=round(float(s.mean()), 4),
        tuned_folds=np.round(s, 4).tolist(),
        fold_wins_vs_baseline=int((s > base_scores).sum()),
        learning_rate=lr_pick,
        mean_trees=int(np.mean(tr)),
        params=best,
    )
    log(f"kept lr {lr_pick}: {s.mean():.4f} vs baseline {base_scores.mean():.4f} "
        f"({s.mean() - base_scores.mean():+.4f}), wins {(s > base_scores).sum()} of {N_SPLITS} folds")
    log(f"elapsed {(time.time() - t0) / 60:.1f} min")


# ---------- blend ----------
def rank(p):
    return rankdata(p) / len(p)


def fold_idx(seed):
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    return [val for _, val in folds.split(np.zeros(len(y)), y)]


def blend_scores(oof_a, oof_b, w, idx):
    pred = w * rank(oof_a) + (1 - w) * rank(oof_b)
    return np.array([roc_auc_score(y.iloc[i], pred[i]) for i in idx])


log("\n===== blend (rank average, LightGBM weight w) =====")
idx42 = fold_idx(SEED)
L, C = results["lgbm"], results["catboost"]
base_blend = blend_scores(L["base_oof"], C["base_oof"], 0.5, idx42)
log(f"baseline 50/50 blend: {fmt(base_blend)}")

weights = np.round(np.arange(0, 1.0001, 0.05), 2)
table = {w: blend_scores(L["oof"], C["oof"], w, idx42) for w in weights}
w_best = max(table, key=lambda w: table[w].mean())
for w in [0.0, 0.25, 0.5, 0.75, 1.0, w_best]:
    log(f"tuned blend w={w:.2f}: {table[w].mean():.4f}")
log(f"best weight {w_best:.2f}: {fmt(table[w_best])}, wins vs baseline blend "
    f"{(table[w_best] > base_blend).sum()} of {N_SPLITS}")
summary["blend"] = dict(
    baseline_50_50_auc=round(float(base_blend.mean()), 4),
    best_lgbm_weight=float(w_best),
    best_auc=round(float(table[w_best].mean()), 4),
    best_folds=np.round(table[w_best], 4).tolist(),
    tuned_50_50_auc=round(float(table[0.5].mean()), 4),
    auc_by_weight={f"{w:.2f}": round(float(table[w].mean()), 4) for w in weights},
)


# ---------- re-check on a different fold split ----------
log(f"\n===== re-check on fold split seed {CHECK_SEED} =====")
check = {}
for name in ["lgbm", "catboost"]:
    r = results[name]
    b_s, b_o, _ = run_cv(name, CURRENT[name], FINAL_LR[name], seed=CHECK_SEED)
    t_s, t_o, _ = run_cv(name, r["params"], r["lr"], seed=CHECK_SEED)
    check[name] = (b_o, t_o)
    log(f"{name}: baseline {b_s.mean():.4f}, tuned {t_s.mean():.4f} "
        f"({t_s.mean() - b_s.mean():+.4f}), wins {(t_s > b_s).sum()} of {N_SPLITS}")
    summary[name]["check_seed7"] = dict(baseline=round(float(b_s.mean()), 4),
                                        tuned=round(float(t_s.mean()), 4),
                                        wins=int((t_s > b_s).sum()))
idx7 = fold_idx(CHECK_SEED)
b7 = blend_scores(check["lgbm"][0], check["catboost"][0], 0.5, idx7)
t7 = blend_scores(check["lgbm"][1], check["catboost"][1], w_best, idx7)
log(f"blend: baseline 50/50 {b7.mean():.4f}, tuned w={w_best:.2f} {t7.mean():.4f} "
    f"({t7.mean() - b7.mean():+.4f}), wins {(t7 > b7).sum()} of {N_SPLITS}")
summary["blend"]["check_seed7"] = dict(baseline=round(float(b7.mean()), 4),
                                       tuned=round(float(t7.mean()), 4),
                                       wins=int((t7 > b7).sum()))


# ---------- final models on all rows ----------
log("\n===== final models =====")
for name in ["lgbm", "catboost"]:
    r = results[name]
    X_tr, X_te = DATA[name]
    n_trees = int(np.mean(r["trees"]) * 1.1)
    model = make(name, r["params"], r["lr"], n_estimators=n_trees)
    model.fit(X_tr, y)
    test_pred = model.predict_proba(X_te)[:, 1]
    tag = f"{name}_quick" if QUICK else f"{name}_tuned"
    np.save(PRED_OUT / f"{tag}_oof.npy", r["oof"])
    np.save(PRED_OUT / f"{tag}_test.npy", test_pred)
    summary[name]["final_trees"] = n_trees
    log(f"{name}: trained {n_trees} trees on all rows, saved {PRED_OUT}/{tag}_oof.npy and _test.npy")

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
log(f"\nsaved {OUT}/summary.json, total {(time.time() - t0) / 60:.1f} min")
