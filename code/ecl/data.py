"""Loan-level random draws, panel loading and the model families."""
from __future__ import annotations

import hashlib
import itertools

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from ecl import CATEGORICAL, DERIVED, FEATURES, MONOTONE, NUMERIC, SEED

PANEL = DERIVED / "panel"
KEY = SEED.to_bytes(8, "little")


def loan_draws() -> pl.DataFrame:
    """Three independent uniform draws per loan from a keyed BLAKE2b hash of
    its identifier, so a loan's assignment does not depend on which other
    loans are in the data: u_cap (row cap), u_split (held-out / early-stop /
    fit) and u_port (portfolio scoring sample)."""
    out = DERIVED / "loan_draws.parquet"
    if out.exists():
        return pl.read_parquet(out)
    ids = pl.read_parquet(DERIVED / "loans.parquet", columns=["loan_id"])["loan_id"].to_list()
    raw = np.frombuffer(b"".join(hashlib.blake2b(i.encode(), digest_size=24, key=KEY).digest() for i in ids),
                        dtype="<u8").reshape(-1, 3)
    u = (raw >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    df = pl.DataFrame({"loan_id": ids, "u_cap": u[:, 0], "u_split": u[:, 1], "u_port": u[:, 2]})
    df.write_parquet(out)
    return df


def scan_panel() -> pl.LazyFrame:
    return pl.scan_parquet(PANEL / "*" / "*.parquet", hive_partitioning=True)


def to_matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).cast(pl.Float64).to_numpy()


# ---- model families -------------------------------------------------------

def lgb_grid():
    """The design's 64 LightGBM configurations, as in Paper 9: 15 or 63 leaves
    x four row/column subsampling settings x five seeds for subsampled
    settings and one for the deterministic setting, unconstrained and
    monotone."""
    for mono, leaves in itertools.product([False, True], [15, 63]):
        for ff, bf in [(0.8, 0.8), (0.8, 1.0), (1.0, 0.8), (1.0, 1.0)]:
            for s in ([1] if (ff, bf) == (1.0, 1.0) else [1, 2, 3, 4, 5]):
                yield {"family": "lgbm_mono" if mono else "lgbm", "num_leaves": leaves,
                       "feature_fraction": ff, "bagging_fraction": bf, "seed": s}


def lgb_params(cfg: dict, threads: int, monotone: bool) -> dict:
    p = {"objective": "binary", "learning_rate": 0.1, "num_leaves": cfg["num_leaves"],
         "min_child_samples": 100, "feature_fraction": cfg["feature_fraction"],
         "bagging_fraction": cfg["bagging_fraction"], "bagging_freq": 1 if cfg["bagging_fraction"] < 1 else 0,
         "seed": cfg["seed"], "deterministic": True, "force_col_wise": True,
         "num_threads": threads, "verbose": -1}
    if monotone:
        p["monotone_constraints"] = [MONOTONE.get(f, 0) for f in FEATURES]
    return p


def fit_lgb(cfg, X, y, w, Xes, yes, threads):
    """Early stopping on the early-stop loans' log loss (50 rounds patience,
    at most 3,000 rounds), as in Paper 9."""
    cat = list(CATEGORICAL)
    dtr = lgb.Dataset(X, y, weight=w, feature_name=FEATURES, categorical_feature=cat, free_raw_data=False)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    return lgb.train(lgb_params(cfg, threads, cfg["family"] == "lgbm_mono"), dtr, num_boost_round=3000,
                     valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])


class SplineLogit:
    """Logistic regression on cubic splines of the numeric inputs (five
    quantile knots, constant beyond the outer knots, as in Paper 9) and
    one-hot categoricals. A missing numeric input is set to its training
    median and flagged with an indicator column; a missing category gets
    its own column."""

    def _basis(self, X, fit=False):
        cols = []
        for j, f in enumerate(FEATURES):
            x = X[:, [j]]
            miss = np.isnan(x)
            if f in CATEGORICAL:
                xc = np.where(miss, -1.0, x)
                if fit:
                    self.enc[f] = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(xc)
                cols.append(self.enc[f].transform(xc))
                continue
            if fit:
                self.med[f] = float(np.nanmedian(x))
                self.has_miss[f] = bool(miss.any())
            xf = np.where(miss, self.med[f], x)
            if fit:
                self.enc[f] = SplineTransformer(n_knots=5, degree=3, knots="quantile",
                                                extrapolation="constant").fit(xf)
            cols.append(self.enc[f].transform(xf))
            if self.has_miss[f]:
                cols.append(miss.astype(np.float64))
        return np.hstack(cols)

    def fit(self, X, y, w=None):
        self.enc, self.med, self.has_miss = {}, {}, {}
        B = self._basis(X, fit=True)
        self.mu, self.sd = B.mean(axis=0), B.std(axis=0) + 1e-12
        self.lr = LogisticRegression(max_iter=3000, C=1.0, tol=1e-6).fit((B - self.mu) / self.sd, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.lr.predict_proba((self._basis(X) - self.mu) / self.sd)[:, 1]


def predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, SplineLogit):
        return model.predict(X)
    return model.predict(X, num_iteration=model.best_iteration)


assert set(MONOTONE) <= set(NUMERIC)
