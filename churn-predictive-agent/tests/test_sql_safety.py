import sqlite3

import pytest

from benchmarks.eval_copilot import ATTACK_SQL, CONTROL_SQL
from src.sql_safety import ReadOnlyDatabase, UnsafeSQLError, validate_sql
from tests.conftest import snapshot_tables


@pytest.mark.parametrize("sql", ATTACK_SQL)
def test_attacks_are_blocked(small_db, sql):
    before = snapshot_tables(small_db)
    db = ReadOnlyDatabase(small_db)
    with pytest.raises((UnsafeSQLError, sqlite3.Error)):
        db.execute(sql)
    db.close()
    assert snapshot_tables(small_db) == before


@pytest.mark.parametrize("sql,params", CONTROL_SQL)
def test_legitimate_queries_are_allowed(small_db, sql, params):
    db = ReadOnlyDatabase(small_db)
    db.execute(sql, params)
    db.close()


@pytest.mark.parametrize("sql,ok", [
    ("SELECT 1; SELECT 2", False), ("  ", False), ("EXPLAIN SELECT 1", False),
    ("SELECT 1 ;; ", True),  # empty statements after the only SELECT are harmless
    ("select * from t; --", True),  # so is a trailing comment
])
def test_validator_edge_cases(sql, ok):
    assert validate_sql(sql).ok is ok


def test_recursive_ctes_are_denied(small_db):
    db = ReadOnlyDatabase(small_db)
    with pytest.raises(UnsafeSQLError, match="authorizer"):
        db.execute("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r")
    db.close()


def test_authorizer_blocks_what_the_static_check_allows(small_db):
    """sqlite_master and non-allow-listed tables pass the token check but are denied by SQLite itself."""
    db = ReadOnlyDatabase(small_db, allowed_tables=["customers"])
    assert validate_sql("SELECT * FROM payments").ok
    with pytest.raises(UnsafeSQLError, match="authorizer"):
        db.execute("SELECT * FROM payments")
    with pytest.raises(UnsafeSQLError, match="authorizer"):
        db.execute("SELECT name FROM sqlite_master")
    db.close()


def test_connection_is_read_only_even_without_the_validator(small_db):
    db = ReadOnlyDatabase(small_db)
    db._con.set_authorizer(None)  # simulate a bypass of layers 1 and 2
    with pytest.raises(sqlite3.Error):
        db._con.execute("DELETE FROM customers")
    db.close()


def test_parameters_are_bound_not_spliced(small_db):
    db = ReadOnlyDatabase(small_db)
    r = db.execute("SELECT COUNT(*) FROM customers WHERE plan = ?", ["Premium' OR '1'='1"])
    assert r.rows == [(0,)]
    with pytest.raises(UnsafeSQLError, match="placeholders"):
        db.execute("SELECT COUNT(*) FROM customers WHERE plan = ?", [])
    db.close()


def test_timeout_and_row_cap(small_db):
    db = ReadOnlyDatabase(small_db, timeout_s=0.3, max_rows=10)
    with pytest.raises(TimeoutError):
        db.execute("SELECT COUNT(*) FROM usage_monthly a, usage_monthly b")
    r = db.execute("SELECT customer_id FROM customers")
    assert len(r.rows) == 10 and r.truncated
    db.close()
