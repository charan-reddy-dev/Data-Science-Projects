import sqlite3
from pathlib import Path

import pytest

from src.build_db import build
from src import churn_model


@pytest.fixture(scope="session")
def small_db(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("db")
    db, oracle = d / "churn.db", d / "oracle.csv"
    build(db, oracle, n=1500, seed=7)
    result = churn_model.train_and_score(db, oracle)
    churn_model.write_scores(result, db)
    return db


@pytest.fixture(scope="session")
def small_model(small_db) -> churn_model.TrainResult:
    return churn_model.train_and_score(small_db, small_db.parent / "oracle.csv")


def snapshot_tables(db: Path) -> dict:
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*), TOTAL(rowid) FROM {t}").fetchone()
                for t in ("customers", "support_tickets", "payments", "usage_monthly", "churn_scores")}
    finally:
        con.close()
