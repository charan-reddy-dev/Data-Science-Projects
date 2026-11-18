import numpy as np
import pytest

from src import churn_model
from src.churn_model import OUTCOME_COLUMNS, assert_no_outcome_features, load_features, make_lgbm, shap_values


def test_outcome_columns_are_never_features(small_db):
    X, y, ids = load_features(small_db)
    assert not OUTCOME_COLUMNS & set(X.columns)
    with pytest.raises(ValueError):
        assert_no_outcome_features(list(X.columns) + ["churned"])


def test_behavioural_tables_stop_at_the_snapshot(small_db):
    import sqlite3

    con = sqlite3.connect(small_db)
    assert con.execute("SELECT MAX(opened_at) FROM support_tickets").fetchone()[0] <= "2025-12-31"
    assert con.execute("SELECT MAX(due_date) FROM payments").fetchone()[0] <= "2025-12-31"
    assert con.execute("SELECT MAX(month) FROM usage_monthly").fetchone()[0] <= "2025-12"
    assert con.execute("SELECT MIN(churn_date) FROM customers").fetchone()[0] > "2025-12-31"
    con.close()


def test_shap_values_match_lightgbm_treeshap(small_db):
    X, y, _ = load_features(small_db)
    model = make_lgbm("c_stumps").fit(X, y)
    ours = shap_values(model, X)
    native = model.predict(X, pred_contrib=True)  # LightGBM's own TreeSHAP: features + bias column
    np.testing.assert_allclose(ours, native[:, :-1], atol=1e-6)
    np.testing.assert_allclose(ours.sum(axis=1) + native[:, -1], model.predict(X, raw_score=True), atol=1e-6)


def test_scores_tiers_and_drivers(small_model):
    s, d = small_model.scores, small_model.drivers
    assert s["churn_probability"].between(0, 1).all()
    assert s["risk_tier"].value_counts().to_dict() == {"Low": 1050, "Medium": 300, "High": 150}
    high_min = s.loc[s.risk_tier == "High", "churn_probability"].min()
    assert high_min >= s.loc[s.risk_tier != "High", "churn_probability"].max()
    assert (d.groupby("customer_id")["rank"].apply(list) == [[1, 2, 3]] * len(s)).all()
    first = d[d["rank"] == 1].set_index("customer_id")["shap_value"]
    second = d[d["rank"] == 2].set_index("customer_id")["shap_value"]
    assert (first >= second).all()


def test_evaluation_is_sane(small_model):
    e = small_model.evaluation
    assert e["test"]["lightgbm"]["roc_auc"] > 0.65
    assert 0.35 < e["label_shuffle_control_test_roc_auc"]["mean"] < 0.65
    assert e["cv_train_5fold"]["lightgbm_selected_config"] in churn_model.LGBM_GRID
