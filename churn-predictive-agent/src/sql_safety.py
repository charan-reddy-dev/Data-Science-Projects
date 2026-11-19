"""Read-only SQL safety: a static validator plus a sandboxed executor.

Three independent layers; a query must pass all of them:

1. `validate_sql` (static, sqlparse tokens): exactly one statement, of type SELECT (optionally WITH ... SELECT);
   no data-modifying or schema keywords anywhere in the token stream (DML other than SELECT, DDL, ATTACH,
   PRAGMA, VACUUM, ...); no dangerous functions (load_extension, readfile, writefile, ...). String literals and
   comments are separate tokens, so `WHERE category = 'drop'` is fine and `/* */; DROP` is still caught.
2. SQLite authorizer (`ReadOnlyDatabase`): SQLite itself reports every table, column and function a prepared
   statement touches. Only SELECT, reads of allow-listed tables and allow-listed functions are permitted;
   everything else (including recursive CTEs) is denied before execution.
3. The connection is opened read-only (`mode=ro` URI) with `PRAGMA query_only = ON`, so even a statement that
   slipped past both checks could not write.

Execution also has a wall-clock budget (progress handler) and a row cap.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import sqlparse
from sqlparse import tokens as T

MAX_SQL_CHARS = 5000
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE", "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "GRANT", "REVOKE", "COPY", "CALL", "EXEC",
    "EXECUTE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "INTO",
}
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "sqlite_offset"}
SAFE_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "total", "group_concat", "round", "abs", "coalesce", "ifnull", "nullif",
    "substr", "substring", "lower", "upper", "length", "trim", "ltrim", "rtrim", "replace", "instr", "cast",
    "date", "datetime", "strftime", "julianday", "iif", "printf", "format", "like", "glob", "typeof",
}


class UnsafeSQLError(ValueError):
    pass


@dataclass(frozen=True)
class Validation:
    ok: bool
    reason: str = ""


def _significant(stmt) -> list:
    return [t for t in stmt.flatten() if not t.is_whitespace and t.ttype not in T.Comment and t.ttype not in (T.Comment.Single, T.Comment.Multiline)]


def validate_sql(sql: str) -> Validation:
    """Static check. Returns Validation(ok, reason); never executes anything."""
    if not isinstance(sql, str) or not sql.strip():
        return Validation(False, "empty query")
    if len(sql) > MAX_SQL_CHARS:
        return Validation(False, f"query longer than {MAX_SQL_CHARS} characters")
    if "\x00" in sql:
        return Validation(False, "NUL byte in query")

    statements = [s for s in sqlparse.parse(sql) if any(t.ttype is not T.Punctuation for t in _significant(s))]
    if len(statements) != 1:
        return Validation(False, f"expected exactly 1 statement, found {len(statements)}")
    stmt = statements[0]
    tokens = _significant(stmt)
    semicolons = [i for i, t in enumerate(tokens) if t.ttype is T.Punctuation and t.value == ";"]
    if semicolons and semicolons != [len(tokens) - 1]:
        return Validation(False, "statement separator inside the query")

    first = tokens[0]
    if not (first.ttype in T.Keyword.DML and first.value.upper() == "SELECT") and not (first.ttype in T.Keyword.CTE):
        return Validation(False, f"only SELECT queries are allowed (got {first.value.upper()!r})")
    if stmt.get_type() != "SELECT":
        return Validation(False, f"only SELECT queries are allowed (statement type {stmt.get_type()})")

    for i, tok in enumerate(tokens):
        upper = tok.value.upper()
        if tok.ttype in T.Keyword.DML and upper != "SELECT":
            return Validation(False, f"forbidden keyword {upper}")
        if tok.ttype in T.Keyword.DDL:
            return Validation(False, f"forbidden keyword {upper}")
        if tok.ttype in T.Keyword and upper in FORBIDDEN_KEYWORDS:
            return Validation(False, f"forbidden keyword {upper}")
        if tok.ttype in T.Name and tok.value.lower() in FORBIDDEN_FUNCTIONS:
            return Validation(False, f"forbidden function {tok.value}")
        if tok.ttype in T.Name.Builtin and upper in FORBIDDEN_KEYWORDS:
            return Validation(False, f"forbidden keyword {upper}")
    return Validation(True)


def count_placeholders(sql: str) -> int:
    return sum(1 for t in sqlparse.parse(sql)[0].flatten() if t.ttype in T.Name.Placeholder and t.value == "?")


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


@dataclass
class ReadOnlyDatabase:
    """A read-only, authorizer-guarded connection to a SQLite file."""

    path: Path
    allowed_tables: Optional[Sequence[str]] = None
    timeout_s: float = 3.0
    max_rows: int = 1000
    denied: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self._con.execute("PRAGMA query_only = ON")
        if self.allowed_tables is None:
            self.allowed_tables = [r[0] for r in self._con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        self._allowed = {t.lower() for t in self.allowed_tables}
        self._con.set_authorizer(self._authorize)

    def _authorize(self, action, arg1, arg2, db_name, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 and arg1.lower() in self._allowed:
                return sqlite3.SQLITE_OK
            self.denied.append(f"read {arg1}.{arg2}")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            if arg2 and arg2.lower() in SAFE_FUNCTIONS:
                return sqlite3.SQLITE_OK
            self.denied.append(f"function {arg2}")
            return sqlite3.SQLITE_DENY
        # Everything else is denied, including SQLITE_RECURSIVE: recursive CTEs are an easy way to burn CPU and
        # are not needed for analytics over this schema.
        self.denied.append(f"action {action} ({arg1}, {arg2})")
        return sqlite3.SQLITE_DENY

    def execute(self, sql: str, params: Sequence[Any] = ()) -> QueryResult:
        check = validate_sql(sql)
        if not check.ok:
            raise UnsafeSQLError(check.reason)
        if count_placeholders(sql) != len(params):
            raise UnsafeSQLError(f"query has {count_placeholders(sql)} placeholders but {len(params)} parameters")
        if any(not isinstance(p, (str, int, float)) or isinstance(p, bool) for p in params):
            raise UnsafeSQLError("parameters must be str, int or float")
        deadline = time.perf_counter() + self.timeout_s
        self._con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        start = time.perf_counter()
        denied_before = len(self.denied)
        try:
            cur = self._con.execute(sql, tuple(params))
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.DatabaseError as exc:
            if len(self.denied) > denied_before:
                raise UnsafeSQLError(f"blocked by SQLite authorizer: {self.denied[-1]}") from exc
            if "interrupted" in str(exc):
                raise TimeoutError(f"query exceeded {self.timeout_s} s") from exc
            raise
        finally:
            self._con.set_progress_handler(None, 0)
        columns = [d[0] for d in cur.description or []]
        return QueryResult(columns, rows[: self.max_rows], len(rows) > self.max_rows, (time.perf_counter() - start) * 1000)

    def close(self) -> None:
        self._con.close()
