"""Shared feature engineering for every model notebook.

Workflow:
1. Ideas start in CANDIDATES below.
2. The "Feature experiments" section of 03_lightgbm.ipynb tests each one
   against the current features on the same 5 folds.
3. Move a candidate into add_features() only if it improves mean AUC
   and wins in most folds, ideally on more than one fold split.
4. Then rerun all three model notebooks and the blend.

Results log (10 folds from two fold splits, LightGBM / CatBoost change in AUC):
- Kept: unit fixes + debt_to_income, about +0.0010 / +0.0011, 8 of 10 folds on each.
- Dropped: rent_to_income, age_at_move_in, unknown_count. No gain together
  (+0.0001 / -0.0005 against the raw columns).
- Rejected: category combinations (employment x company size, channel x age band)
  made both tree models worse, by up to -0.0042.
- Rejected (Nov 10, tuned LightGBM, seed 42 / seed 7): peer comparisons, the log gap
  between a value and its peer group's median (PEER_CANDIDATES). Every one scored
  between -0.0021 and +0.0006, winning at most 5 of 10 folds; all 7 together
  -0.0018 / -0.0005. Declared incomes cluster on round numbers, so many groups
  share the same median and the gaps add little beyond raw income.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from paths import DATA

INCOME ="Declared Annual Income"
DEBT = "Declared Borrowing Amount"
N_LOANS = "Declared Number of Borrowings"
LIMIT = "Desired Credit Limit"
RENT = "Rent Burden"
AGE = "Age (in months)"
RESIDENCE = "Residence Period (in months)"
UNKNOWN_IS_ZERO = ["Residence Ownership Status", "Insurance Type", "Business Entity Type"]

# Values above these look like unit errors: divided back down, they match
# normal incomes and rents.
INCOME_ERROR_ABOVE = 1e8  # ¥100 million a year; such values look 10,000 times too big
RENT_ERROR_ABOVE = 1e6    # ¥1 million a month; such values look 1,000 times too big


def fix_units(df):
    """Divide incomes and rents that look entered in the wrong unit."""
    df = df.copy()
    big_income = df[INCOME] > INCOME_ERROR_ABOVE
    df.loc[big_income, INCOME] = df.loc[big_income, INCOME] / 1e4
    big_rent = df[RENT] > RENT_ERROR_ABOVE
    df.loc[big_rent, RENT] = df.loc[big_rent, RENT] / 1e3
    return df


def safe_income(df):
    """Income with a floor of 1, so dividing by it never divides by zero."""
    return df[INCOME].clip(lower=1)


# ---------- Peer comparisons ----------
# "How does this applicant compare with people like them?"
# Each peer feature compares one value, such as income, with the median of a
# peer group, such as everyone in the same occupation.
#
# The medians are worked out ONCE, from the train and test inputs together,
# and the same numbers are used for both tables. Computing them inside each
# add_features() call would give train and test different medians.
# Only input columns are read, never the default flag, so no answer leaks.

DATA_FILES = [DATA / "train_dataset.csv", DATA / "test_without_gt.csv"]
MIN_PEERS = 30  # groups smaller than this use the overall median instead

PEER_GROUPS = {
    "occupation": lambda df: df["Occupation Category"],
    "industry": lambda df: df["Industry Category"],
    "company_size": lambda df: df["Company Size Category"],
    "employment": lambda df: df["Employment Type Category"],
    "age_band": lambda df: df[AGE] // 60,  # 5-year age bands
}
PEER_COLUMNS = [
    INCOME, DEBT, LIMIT, RENT, AGE, "Occupation Category", "Industry Category",
    "Company Size Category", "Employment Type Category",
]


@lru_cache(maxsize=None)
def peer_reference():
    """Train and test inputs together, with units fixed. Read once per session."""
    frames = [pd.read_csv(path, usecols=PEER_COLUMNS) for path in DATA_FILES]
    return fix_units(pd.concat(frames, ignore_index=True))


@lru_cache(maxsize=None)
def peer_medians(value, group):
    """Median of `value` in each peer group, and the overall median."""
    ref = peer_reference()
    stats = ref.groupby(PEER_GROUPS[group](ref))[value].agg(["median", "size"])
    overall = ref[value].median()
    return stats["median"].where(stats["size"] >= MIN_PEERS, overall), overall


def peer_gap(df, value, group):
    """How far a row's value is from its peer group's median, on a log scale.

    0 means typical for the group, about +0.7 means double, about -0.7 means half.
    The log scale copes with zeros, such as people with no existing debt.
    """
    medians, overall = peer_medians(value, group)
    typical = PEER_GROUPS[group](df).map(medians).fillna(overall)
    return np.log1p(df[value]) - np.log1p(typical)


# Each candidate takes the feature table and returns one new column.
# Remember: by the time candidates are tested in the notebook, X has already
# been through add_features(), so incomes and rents are already fixed.
CANDIDATES = {
    "debt_to_income": lambda df: df[DEBT] / safe_income(df),
    "total_ask_to_income": lambda df: (df[DEBT] + df[LIMIT]) / safe_income(df),
    "limit_to_debt": lambda df: df[LIMIT] / (df[DEBT] + 1e4),
    "limit_to_income": lambda df: df[LIMIT] / safe_income(df),
    "rent_to_income": lambda df: df[RENT] * 12 / safe_income(df),
    "avg_loan": lambda df: df[DEBT] / df[N_LOANS].clip(lower=1),
    "age_at_move_in": lambda df: df[AGE] - df[RESIDENCE],
    "residence_share": lambda df: df[RESIDENCE] / df[AGE],
    "unknown_count": lambda df: (df[UNKNOWN_IS_ZERO] == 0).sum(axis=1),
}

# Peer candidates: (value column, peer group).
PEER_CANDIDATES = {
    "income_vs_occupation": (INCOME, "occupation"),
    "income_vs_industry": (INCOME, "industry"),
    "income_vs_company_size": (INCOME, "company_size"),
    "income_vs_employment": (INCOME, "employment"),
    "income_vs_age": (INCOME, "age_band"),
    "debt_vs_age": (DEBT, "age_band"),
    "limit_vs_age": (LIMIT, "age_band"),
}
for _name, (_value, _group) in PEER_CANDIDATES.items():
    # v=_value, g=_group freeze this loop's values. Without them, every lambda
    # would use the last pair from the loop.
    CANDIDATES[_name] = lambda df, v=_value, g=_group: peer_gap(df, v, g)


def add_features(df):
    """Features that passed testing. Every model notebook calls this.

    To keep a new candidate, add a line like this one:
        df["total_ask_to_income"] = CANDIDATES["total_ask_to_income"](df)
    """
    df = fix_units(df)
    df["debt_to_income"] = CANDIDATES["debt_to_income"](df)
    return df
