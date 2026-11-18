"""Gradient-boosted churn model with per-customer SHAP explanations.

    python -m src.churn_model        # trains, evaluates, writes churn_scores / churn_drivers / churn_feature_importance

* Features are built in SQL from data up to the snapshot (2025-12-31); the outcome columns
  (customers.churned, customers.churn_date) are never features (`assert_no_outcome_features`).
* Evaluation: stratified 80/20 split (seed 42). LightGBM settings are chosen from a small fixed grid by 5-fold
  stratified CV on the 80%; a logistic-regression baseline is evaluated the same way. Both are refitted on the
  80% and scored once on the 20% test set.
* Scores written to the database are cross-fitted: every customer is scored (and explained with SHAP) by a
  model trained on the other 4 folds, so cohort statistics such as "observed churn rate of the High tier" are
  out-of-sample.
* Risk tiers: High = top 10% by predicted probability, Medium = next 20%, Low = remaining 70%.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.build_db import DB_PATH, ORACLE_PATH

SEED = 42
MODEL_VERSION = "lgbm-v1"
OUTCOME_COLUMNS = frozenset({"churned", "churn_date"})
CATEGORICAL = ("region", "segment", "contract", "plan", "payment_method")
TIERS = (("High", 0.10), ("Medium", 0.20), ("Low", 0.70))

FEATURE_SQL = """
WITH t AS (
    SELECT customer_id,
           COUNT(*) AS tickets_12m,
           SUM(opened_at >= '2025-10-01') AS tickets_90d,
           SUM(opened_at >= '2025-10-01' AND severity = 'high') AS high_severity_tickets_90d,
           SUM(opened_at >= '2025-10-01' AND resolved_hours > 48) AS slow_tickets_90d,
           AVG(resolved_hours) AS avg_resolution_hours_12m,
           MIN(csat) AS min_csat_12m
    FROM support_tickets WHERE opened_at <= '2025-12-31' GROUP BY customer_id
), u AS (
    SELECT customer_id,
           AVG(CASE WHEN month >= '2025-10' THEN data_gb END) AS data_gb_last3,
           AVG(CASE WHEN month BETWEEN '2025-04' AND '2025-09' THEN data_gb END) AS data_gb_prior6,
           AVG(CASE WHEN month >= '2025-10' THEN call_minutes END) AS call_minutes_last3,
           SUM(CASE WHEN month >= '2025-10' THEN outage_minutes END) AS outage_minutes_90d
    FROM usage_monthly WHERE month <= '2025-12' GROUP BY customer_id
), p AS (
    SELECT customer_id,
           SUM(status = 'failed' AND due_date >= '2025-07-01') AS failed_payments_6m,
           SUM(status = 'late' AND due_date >= '2025-07-01') AS late_payments_6m
    FROM payments WHERE due_date <= '2025-12-31' GROUP BY customer_id
)
SELECT c.customer_id, c.region, c.segment, c.age, c.contract, c.plan, c.payment_method, c.paperless_billing,
       c.monthly_charges, c.tenure_months,
       COALESCE(t.tickets_12m, 0) AS tickets_12m, COALESCE(t.tickets_90d, 0) AS tickets_90d,
       COALESCE(t.high_severity_tickets_90d, 0) AS high_severity_tickets_90d,
       COALESCE(t.slow_tickets_90d, 0) AS slow_tickets_90d,
       t.avg_resolution_hours_12m, t.min_csat_12m,
       u.data_gb_last3, u.data_gb_prior6, u.data_gb_last3 / u.data_gb_prior6 AS usage_trend,
       u.call_minutes_last3, u.outage_minutes_90d,
       COALESCE(p.failed_payments_6m, 0) AS failed_payments_6m, COALESCE(p.late_payments_6m, 0) AS late_payments_6m,
       c.churned
FROM customers c LEFT JOIN t USING (customer_id) LEFT JOIN u USING (customer_id) LEFT JOIN p USING (customer_id)
ORDER BY c.customer_id
"""

FEATURE_DESCRIPTIONS = {
    "region": "service region", "segment": "Consumer or SMB", "age": "account holder age",
    "contract": "contract type", "plan": "plan tier", "payment_method": "payment method",
    "paperless_billing": "paperless billing flag", "monthly_charges": "monthly charge (INR)",
    "tenure_months": "months since signup", "tickets_12m": "support tickets in 2025",
    "tickets_90d": "support tickets in the last 90 days", "high_severity_tickets_90d": "high-severity tickets, last 90 days",
    "slow_tickets_90d": "tickets taking over 48 h to resolve, last 90 days",
    "avg_resolution_hours_12m": "average resolution hours, 2025", "min_csat_12m": "lowest CSAT score in 2025",
    "data_gb_last3": "average monthly data, Oct-Dec", "data_gb_prior6": "average monthly data, Apr-Sep",
    "usage_trend": "data usage Oct-Dec / Apr-Sep", "call_minutes_last3": "average call minutes, Oct-Dec",
    "outage_minutes_90d": "outage minutes, Oct-Dec", "failed_payments_6m": "failed payments, Jul-Dec",
    "late_payments_6m": "late payments, Jul-Dec",
}


def assert_no_outcome_features(columns) -> None:
    leaked = OUTCOME_COLUMNS & set(columns)
    if leaked:
        raise ValueError(f"outcome columns used as features: {sorted(leaked)}")


def load_features(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(FEATURE_SQL, con)
    finally:
        con.close()
    y = df.pop("churned").to_numpy()
    ids = df.pop("customer_id").to_numpy()
    assert_no_outcome_features(df.columns)
    for c in CATEGORICAL:
        df[c] = pd.Categorical(df[c], categories=sorted(df[c].unique()))
    return df, y, ids


# --------------------------------------------------------------------------- models


LGBM_GRID = {  # small, fixed grid; selected by 5-fold CV ROC-AUC on the training split only
    "a_baseline": dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40),
    "b_shallow": dict(n_estimators=500, learning_rate=0.02, num_leaves=7, min_child_samples=80),
    "c_stumps": dict(n_estimators=800, learning_rate=0.01, num_leaves=4, min_child_samples=100),
    "d_regularised": dict(n_estimators=400, learning_rate=0.03, num_leaves=4, min_child_samples=50, reg_lambda=5.0),
}


def make_lgbm(config: str = "a_baseline"):
    from lightgbm import LGBMClassifier

    params = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
    params.update(LGBM_GRID[config])
    return LGBMClassifier(**params, random_state=SEED, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)


def make_logistic(X: pd.DataFrame) -> Pipeline:
    num = [c for c in X.columns if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=5000))])


def metrics(y, p) -> dict:
    order = np.argsort(-p, kind="stable")
    top = order[: max(1, len(p) // 10)]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "top_decile_churn_rate": round(float(y[top].mean()), 4),
        "top_decile_lift": round(float(y[top].mean() / y.mean()), 3),
        "n": int(len(y)), "churners": int(y.sum()),
    }


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values for the positive class, shape (n_rows, n_features), in log-odds units."""
    import warnings

    import shap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):  # older/newer shap: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[1] if values.shape[0] == 2 else values[..., 1]
    if values.shape != X.shape:
        raise ValueError(f"unexpected SHAP shape {values.shape} for X {X.shape}")
    return values


# --------------------------------------------------------------------------- training and scoring


@dataclass
class TrainResult:
    evaluation: dict
    scores: pd.DataFrame
    drivers: pd.DataFrame
    importance: pd.DataFrame


def train_and_score(db_path: Path = DB_PATH, oracle_path: Path = ORACLE_PATH) -> TrainResult:
    X, y, ids = load_features(db_path)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, ytr, Xte, yte = X.iloc[idx_train], y[idx_train], X.iloc[idx_test], y[idx_test]

    cv = {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))

    def cv_auc(factory) -> dict:
        s = [roc_auc_score(ytr[te], factory().fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[te])[:, 1]) for tr, te in folds]
        return {"roc_auc_mean": round(float(np.mean(s)), 4), "roc_auc_std": round(float(np.std(s)), 4)}

    grid = {name: cv_auc(lambda name=name: make_lgbm(name)) for name in LGBM_GRID}
    config = max(grid, key=lambda k: grid[k]["roc_auc_mean"])
    cv["lightgbm_grid"] = grid
    cv["lightgbm_selected_config"] = config
    cv["lightgbm"] = grid[config]
    cv["logistic_regression"] = cv_auc(lambda: make_logistic(X))

    lgbm = make_lgbm(config).fit(Xtr, ytr)
    logit = make_logistic(X).fit(Xtr, ytr)
    p_lgbm, p_logit = lgbm.predict_proba(Xte)[:, 1], logit.predict_proba(Xte)[:, 1]
    test = {"lightgbm": metrics(yte, p_lgbm), "logistic_regression": metrics(yte, p_logit)}

    rng = np.random.default_rng(SEED)
    boot = [roc_auc_score(yte[i], p_lgbm[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(500))]
    controls = [roc_auc_score(yte, make_lgbm(config).fit(Xtr, np.random.default_rng([SEED, k]).permutation(ytr)).predict_proba(Xte)[:, 1])
                for k in range(5)]
    oracle = pd.read_csv(oracle_path).set_index("customer_id").loc[ids[idx_test], "oracle_churn_prob"].to_numpy() \
        if Path(oracle_path).exists() else None

    # Cross-fitted scores and SHAP for every customer (each scored by a model that did not see it).
    oof = np.zeros(len(y))
    shap_all = np.zeros(X.shape)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        model = make_lgbm(config).fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        shap_all[te] = shap_values(model, X.iloc[te])

    order = np.argsort(-oof, kind="stable")
    tier = np.empty(len(y), dtype=object)
    start = 0
    for name, share in TIERS:
        stop = len(y) if name == TIERS[-1][0] else start + round(share * len(y))
        tier[order[start:stop]] = name
        start = stop
    scores = pd.DataFrame({"customer_id": ids, "churn_probability": np.round(oof, 6), "risk_tier": tier,
                           "scored_at": "2025-12-31", "model_version": MODEL_VERSION})

    features = list(X.columns)
    driver_rows = []
    for i in range(len(y)):
        for rank, j in enumerate(np.argsort(-shap_all[i], kind="stable")[:3], start=1):
            value = X.iat[i, j]
            driver_rows.append((int(ids[i]), rank, features[j], None if pd.isna(value) else str(value), round(float(shap_all[i, j]), 6)))
    drivers = pd.DataFrame(driver_rows, columns=["customer_id", "rank", "feature", "feature_value", "shap_value"])

    mean_abs = np.abs(shap_all).mean(axis=0)
    high = tier == "High"
    importance = pd.DataFrame({
        "feature": features, "description": [FEATURE_DESCRIPTIONS[f] for f in features],
        "mean_abs_shap": np.round(mean_abs, 6), "mean_shap_high_risk": np.round(shap_all[high].mean(axis=0), 6),
    }).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    cohorts = []
    for name, _ in TIERS:
        m = tier == name
        cohorts.append({"risk_tier": name, "customers": int(m.sum()), "mean_churn_probability": round(float(oof[m].mean()), 4),
                        "observed_churn_rate": round(float(y[m].mean()), 4), "share_of_all_churners": round(float(y[m].sum() / y.sum()), 4)})

    evaluation = {
        "data": {"customers": int(len(y)), "churn_rate": round(float(y.mean()), 4), "features": features,
                 "train": int(len(idx_train)), "test": int(len(idx_test))},
        "hyperparameters": "LightGBM: 4-config grid fixed in code, selected by 5-fold CV ROC-AUC on the training split; "
                           "logistic regression: defaults. The test split is scored once with the selected model.",
        "cv_train_5fold": cv,
        "test": test,
        "test_lightgbm_roc_auc_bootstrap_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "oracle_test": metrics(yte, oracle) if oracle is not None else None,
        "oracle_note": "generator's own churn probability (includes an unobserved factor); ceiling for any model on this data",
        "label_shuffle_control_test_roc_auc": {"draws": [round(float(c), 4) for c in controls],
                                               "mean": round(float(np.mean(controls)), 4)},
        "cross_fitted_all_customers": metrics(y, oof),
        "risk_tiers": cohorts,
        "top_shap_features": importance.head(8)[["feature", "mean_abs_shap", "mean_shap_high_risk"]].to_dict("records"),
    }
    return TrainResult(evaluation, scores, drivers, importance)


SCORE_DDL = """
DROP TABLE IF EXISTS churn_scores;
DROP TABLE IF EXISTS churn_drivers;
DROP TABLE IF EXISTS churn_feature_importance;
CREATE TABLE churn_scores (customer_id INTEGER PRIMARY KEY REFERENCES customers(customer_id),
    churn_probability REAL NOT NULL, risk_tier TEXT NOT NULL, scored_at TEXT NOT NULL, model_version TEXT NOT NULL);
CREATE TABLE churn_drivers (customer_id INTEGER NOT NULL REFERENCES customers(customer_id), rank INTEGER NOT NULL,
    feature TEXT NOT NULL, feature_value TEXT, shap_value REAL NOT NULL, PRIMARY KEY (customer_id, rank));
CREATE TABLE churn_feature_importance (rank INTEGER PRIMARY KEY, feature TEXT NOT NULL, description TEXT NOT NULL,
    mean_abs_shap REAL NOT NULL, mean_shap_high_risk REAL NOT NULL);
DELETE FROM data_dictionary WHERE table_name IN ('churn_scores', 'churn_drivers', 'churn_feature_importance');
"""

SCORE_DICTIONARY = [
    ("churn_scores", "customer_id", "Customer identifier"),
    ("churn_scores", "churn_probability", "Model-predicted probability of churning in the 90 days after 2025-12-31 (cross-fitted)"),
    ("churn_scores", "risk_tier", "High (top 10% of probability), Medium (next 20%), Low (remaining 70%)"),
    ("churn_scores", "scored_at", "Snapshot date the score refers to"),
    ("churn_scores", "model_version", "Model identifier"),
    ("churn_drivers", "customer_id", "Customer identifier"),
    ("churn_drivers", "rank", "1 = the feature pushing this customer's churn risk up the most"),
    ("churn_drivers", "feature", "Model feature name (see churn_feature_importance.description)"),
    ("churn_drivers", "feature_value", "The customer's value of that feature"),
    ("churn_drivers", "shap_value", "SHAP contribution to the log-odds of churn"),
    ("churn_feature_importance", "rank", "Rank by mean absolute SHAP value"),
    ("churn_feature_importance", "feature", "Model feature name"),
    ("churn_feature_importance", "description", "Plain-English feature definition"),
    ("churn_feature_importance", "mean_abs_shap", "Mean |SHAP| across all customers"),
    ("churn_feature_importance", "mean_shap_high_risk", "Mean SHAP value among High-tier customers"),
]


def write_scores(result: TrainResult, db_path: Path = DB_PATH) -> None:
    """Trusted batch job: the only writer of the score tables (the copilot connects read-only)."""
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCORE_DDL)
        for table, df in (("churn_scores", result.scores), ("churn_drivers", result.drivers),
                          ("churn_feature_importance", result.importance)):
            con.executemany(f"INSERT INTO {table} ({', '.join(df.columns)}) VALUES ({', '.join('?' * len(df.columns))})",
                            df.astype(object).where(df.notna(), None).itertuples(index=False, name=None))
        con.executemany("INSERT INTO data_dictionary VALUES (?, ?, ?)", SCORE_DICTIONARY)
        con.commit()
    finally:
        con.close()


def main() -> dict:
    import json

    result = train_and_score()
    write_scores(result)
    print(json.dumps({k: result.evaluation[k] for k in ("cv_train_5fold", "test", "oracle_test", "risk_tiers")}, indent=1))
    return result.evaluation


if __name__ == "__main__":
    main()
