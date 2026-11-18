"""Schema-grounded text-to-SQL copilot over data/churn.db.

    python -m src.sql_copilot "What is the churn rate by contract type?"
    python -m src.sql_copilot --provider gemini "..."      # LLM first, rule-based fallback

Pipeline for every question:

    question --> [LLM provider: JSON {sql, params}]  --fails/unsafe-->  [rule-based planner: QuerySpec IR]
                         |                                                       |
                         +------------------> validate_sql + SQLite authorizer + read-only connection
                                                         |
                                                 execute with bound parameters --> rows + summary

* Schema grounding: tables, columns, types, foreign keys, the data_dictionary descriptions and the live
  distinct values of low-cardinality text columns are introspected from the database at start-up and sent to
  the LLM; the rule-based planner builds its vocabulary from the same catalogue.
* Every literal from the question is a bound `?` parameter, never spliced into SQL text.
* LLM output is untrusted: it goes through exactly the same validator and sandbox as everything else, and an
  unsafe or failing LLM query falls back to the planner (recorded in `Answer.attempts`).

Providers (environment): CHURNINTEL_LLM = gemini | openai | groq | none (default: the first provider with a
key among GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY; otherwise offline rules only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx

from src.build_db import DB_PATH
from src.sql_safety import QueryResult, ReadOnlyDatabase, UnsafeSQLError, count_placeholders, validate_sql

MAX_DISTINCT_VALUES = 25
DEFAULT_LIST_LIMIT = 50


# =========================================================================== schema catalogue


@dataclass
class ColumnInfo:
    name: str
    type: str
    description: str = ""
    values: Optional[list[str]] = None  # distinct values for low-cardinality text columns
    minimum: Any = None
    maximum: Any = None


@dataclass
class TableInfo:
    name: str
    columns: dict[str, ColumnInfo]
    row_count: int
    primary_key: tuple[str, ...]
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)

    @property
    def per_customer(self) -> bool:
        """One row per customer (safe to JOIN on customer_id without duplicating rows)."""
        return self.primary_key == ("customer_id",)

    @property
    def has_customer_id(self) -> bool:
        return "customer_id" in self.columns


@dataclass
class SchemaCatalog:
    tables: dict[str, TableInfo]

    @classmethod
    def introspect(cls, db_path: Path = DB_PATH) -> "SchemaCatalog":
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            docs: dict[tuple[str, str], str] = {}
            if "data_dictionary" in names:
                docs = {(t, c): d for t, c, d in con.execute("SELECT table_name, column_name, description FROM data_dictionary")}
            tables = {}
            for t in names:
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                pk = tuple(r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5]))
                fks = [(r[3], r[2], r[4]) for r in con.execute(f'PRAGMA foreign_key_list("{t}")')]
                cols = {}
                for _, name, ctype, *_ in info:
                    col = ColumnInfo(name, ctype.upper(), docs.get((t, name), ""))
                    if col.type == "TEXT":
                        n = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{t}"').fetchone()[0]
                        if 0 < n <= MAX_DISTINCT_VALUES:
                            col.values = [r[0] for r in con.execute(
                                f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL ORDER BY 1')]
                    elif col.type in ("INTEGER", "REAL"):
                        col.minimum, col.maximum = con.execute(f'SELECT MIN("{name}"), MAX("{name}") FROM "{t}"').fetchone()
                    cols[name] = col
                tables[t] = TableInfo(t, cols, con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0], pk, fks)
        finally:
            con.close()
        return cls(tables)

    def prompt(self) -> str:
        lines = []
        for t in self.tables.values():
            if t.name == "data_dictionary":
                continue
            lines.append(f"TABLE {t.name} ({t.row_count} rows; primary key {', '.join(t.primary_key) or 'none'})")
            for c in t.columns.values():
                extra = []
                if c.values is not None:
                    extra.append("values: " + ", ".join(repr(v) for v in c.values))
                elif c.minimum is not None:
                    extra.append(f"range {c.minimum}..{c.maximum}")
                lines.append(f"  {c.name} {c.type}" + (f" -- {c.description}" if c.description else "")
                             + (f" [{'; '.join(extra)}]" if extra else ""))
            for col, ref_t, ref_c in t.foreign_keys:
                lines.append(f"  FOREIGN KEY {col} -> {ref_t}.{ref_c}")
        return "\n".join(lines)


# =========================================================================== LLM providers


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleProvider:
    """POST {base_url}/chat/completions (OpenAI, Groq, vLLM, Ollama, LM Studio, ...)."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    name: str = "openai"
    timeout_s: float = 30.0
    client: Optional[httpx.Client] = None

    def complete(self, system: str, user: str) -> str:
        client = self.client or httpx.Client(timeout=self.timeout_s)
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.name}: unexpected response") from exc


@dataclass
class GeminiProvider:
    api_key: str
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com"
    name: str = "gemini"
    timeout_s: float = 30.0
    client: Optional[httpx.Client] = None

    def complete(self, system: str, user: str) -> str:
        client = self.client or httpx.Client(timeout=self.timeout_s)
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        try:
            r = client.post(f"{self.base_url.rstrip('/')}/v1beta/models/{self.model}:generateContent", json=body,
                            headers={"x-goog-api-key": self.api_key})
        except httpx.HTTPError as exc:
            raise LLMError(f"gemini: {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"gemini: HTTP {r.status_code}: {r.text[:200]}")
        try:
            return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("gemini: no content in response") from exc


def provider_from_env(env: Optional[dict] = None) -> Optional[LLMProvider]:
    env = os.environ if env is None else env
    choice = (env.get("CHURNINTEL_LLM") or "").lower()
    if choice == "none":
        return None
    if choice in ("", "gemini") and env.get("GEMINI_API_KEY"):
        return GeminiProvider(env["GEMINI_API_KEY"], env.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if choice in ("", "openai") and env.get("OPENAI_API_KEY"):
        return OpenAICompatibleProvider(env["OPENAI_API_KEY"], env.get("OPENAI_MODEL", "gpt-4o-mini"),
                                        env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    if choice in ("", "groq") and env.get("GROQ_API_KEY"):
        return OpenAICompatibleProvider(env["GROQ_API_KEY"], env.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                        "https://api.groq.com/openai/v1", name="groq")
    if choice:
        raise ValueError(f"CHURNINTEL_LLM={choice} but no matching API key is set")
    return None


SYSTEM_PROMPT = """You translate analytics questions into ONE read-only SQLite query over the schema below.
Rules:
- Output JSON only: {"sql": "<one SELECT statement>", "params": [<values>]}.
- Exactly one SELECT (WITH ... SELECT allowed). Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH or PRAGMA.
- Use only tables and columns listed in the schema. Join on customer_id.
- Put every literal value taken from the question (text or number) in "params" and write ? in the SQL.
- Use the exact categorical spellings listed in the schema.
- Churn rate means AVG(customers.churned). "High risk" means churn_scores.risk_tier = 'High'.
- If the question asks to modify data or cannot be answered from the schema, return {"sql": "", "params": [], "refusal": "<reason>"}.
"""


def _parse_llm_json(text: str) -> tuple[str, list]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError("no JSON object in model output")
    data = json.loads(m.group(0))
    if data.get("refusal") or not data.get("sql"):
        raise LLMError(f"model refused: {data.get('refusal', 'empty sql')}")
    params = data.get("params") or []
    if not isinstance(params, list):
        raise LLMError("params is not a list")
    return str(data["sql"]), params


# =========================================================================== rule-based planner (offline fallback)


class CannotAnswer(ValueError):
    pass


@dataclass
class QuerySpec:
    """Small relational IR the planner builds and `render` turns into parameterised SQL."""

    base: str
    select: list[tuple[str, str]] = field(default_factory=list)  # (expression, alias)
    joins: list[str] = field(default_factory=list)  # per-customer tables joined on customer_id
    where: list[tuple[str, list]] = field(default_factory=list)  # (fragment with ?, params)
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: Optional[int] = None


COLUMN_SYNONYMS = {
    ("customers", "contract"): ["contract type", "contract"],
    ("customers", "payment_method"): ["payment method", "payment type"],
    ("customers", "monthly_charges"): ["monthly charges", "monthly charge", "monthly bill", "charges"],
    ("customers", "tenure_months"): ["tenure"],
    ("customers", "paperless_billing"): ["paperless billing", "paperless"],
    ("customers", "signup_date"): ["signup date", "sign up date"],
    ("support_tickets", "resolved_hours"): ["resolution time", "resolution hours", "hours to resolve", "time to resolve"],
    ("support_tickets", "csat"): ["csat", "satisfaction score", "satisfaction"],
    ("support_tickets", "category"): ["ticket category", "category"],
    ("support_tickets", "severity"): ["severity"],
    ("payments", "status"): ["payment status"],
    ("payments", "amount"): ["payment amount", "billed amount"],
    ("usage_monthly", "data_gb"): ["data usage", "data used", "data gb"],
    ("usage_monthly", "outage_minutes"): ["outage minutes", "outages"],
    ("usage_monthly", "call_minutes"): ["call minutes", "voice minutes"],
    ("churn_scores", "churn_probability"): ["predicted churn probability", "churn probability", "churn risk score", "risk score"],
    ("churn_scores", "risk_tier"): ["risk tier", "risk"],
    ("churn_drivers", "feature"): ["churn drivers", "churn driver", "drivers", "driver"],
    ("churn_feature_importance", "mean_abs_shap"): ["feature importance", "shap importance"],
}
TABLE_SYNONYMS = {
    "customers": ["customers", "customer", "subscribers", "accounts"],
    "support_tickets": ["support tickets", "support ticket", "tickets", "ticket"],
    "payments": ["payments", "payment", "bills"],
    "usage_monthly": ["usage records", "usage"],
}
DATE_COLUMNS = {"payments": "due_date", "support_tickets": "opened_at", "customers": "signup_date", "usage_monthly": "month"}
ID_COLUMNS = {"customers": "customer_id", "support_tickets": "ticket_id", "payments": "payment_id", "churn_scores": "customer_id"}
WRITE_INTENT = re.compile(r"\b(delete|drop|remove|update|insert|truncate|alter|modify|overwrite|wipe|erase|set .* to)\b")
COMPARATORS = [("more than", ">"), ("greater than", ">"), ("over", ">"), ("above", ">"), ("at least", ">="),
               ("fewer than", "<"), ("less than", "<"), ("under", "<"), ("below", "<"), ("at most", "<=")]
CMP_RE = "|".join(re.escape(p) for p, _ in COMPARATORS)
CMP_OP = dict(COMPARATORS)


def normalize(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Span:
    start: int
    end: int
    kind: str  # concept | column | table | value
    payload: Any

    @property
    def length(self) -> int:
        return self.end - self.start


class RulePlanner:
    """Deterministic NL -> QuerySpec planner grounded in the introspected catalogue."""

    PRIORITY = {"concept": 0, "column": 1, "table": 2, "value": 3}
    CONCEPTS = {
        "churn rate": "churn_rate", "rate of churn": "churn_rate", "attrition rate": "churn_rate",
        "did not churn": "retained", "retained": "retained", "churned": "churned",
        "each month": "per_month", "per month": "per_month", "by month": "per_month", "every month": "per_month",
    }

    def __init__(self, catalog: SchemaCatalog):
        self.catalog = catalog
        self.alias = {}
        for t in catalog.tables:
            a = "".join(w[0] for w in t.split("_"))
            while a in self.alias.values():
                a += "x"
            self.alias[t] = a
        phrases: list[tuple[str, str, Any]] = [(p, "concept", c) for p, c in self.CONCEPTS.items()]
        for (t, c), syns in COLUMN_SYNONYMS.items():
            if t in catalog.tables and c in catalog.tables[t].columns:
                phrases += [(s, "column", (t, c)) for s in syns]
        column_phrases = {p for p, k, _ in phrases if k == "column"}
        for t, info in catalog.tables.items():
            if t == "data_dictionary":
                continue
            for c, col in info.columns.items():
                if not c.endswith("_id") and normalize(c) not in column_phrases:
                    phrases.append((normalize(c), "column", (t, c)))
                    column_phrases.add(normalize(c))
                for v in col.values or []:
                    phrases.append((normalize(v), "value", (t, c, v)))
        for t, syns in TABLE_SYNONYMS.items():
            if t in catalog.tables:
                phrases += [(s, "table", t) for s in syns]
        self.phrases = [(p, k, x) for p, k, x in phrases if p]

    # ---------------------------------------------------------------- span detection

    def spans(self, q: str) -> list[Span]:
        found: list[Span] = []
        for phrase, kind, payload in self.phrases:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q):
                found.append(Span(m.start(), m.end(), kind, payload))
        found.sort(key=lambda s: (-s.length, self.PRIORITY[s.kind], s.start))
        chosen: list[Span] = []
        for s in found:
            if all(s.end <= c.start or s.start >= c.end for c in chosen):
                chosen.append(s)
            elif s.kind == "value" and any(c.start == s.start and c.end == s.end and c.kind == "value" for c in chosen):
                chosen.append(s)  # same phrase, several candidate columns: resolved later
        return sorted(chosen, key=lambda s: (s.start, self.PRIORITY[s.kind]))

    # ---------------------------------------------------------------- planning

    def plan(self, question: str) -> tuple[str, list]:
        q = normalize(question)
        if WRITE_INTENT.search(q):
            raise CannotAnswer("the copilot is read-only and will not modify data")
        spans = self.spans(q)
        columns = [s for s in spans if s.kind == "column"]
        tables = [s for s in spans if s.kind == "table"]
        concepts = [s for s in spans if s.kind == "concept"]
        used_values: set[int] = set()

        def col_after(pos: int, max_gap: int = 12) -> Optional[Span]:
            return next((s for s in columns if 0 <= s.start - pos <= max_gap), None)

        def ref(t: str, c: str) -> str:
            return f"{self.alias[t]}.{c}"

        # --- metric / shape of the query
        select: list[tuple[str, str, str]] = []  # (table, expr, alias)
        group: list[tuple[str, str, str]] = []
        order: list[str] = []
        limit = None
        base: Optional[str] = None
        group_spans: set[int] = set()

        top = re.search(r"\b(top|highest|bottom|lowest)\s+(\d+)\s+(?:[a-z]+\s+){0,2}?by\s+", q)
        common = re.search(r"\bmost (?:common|frequent)\b", q)
        if top:
            s = col_after(top.end())
            if not s:
                raise CannotAnswer("could not tell which column to rank by")
            t, c = s.payload
            ent = next((x.payload for x in tables if x.start < top.end()), t)
            base = ent if ent in ID_COLUMNS else t
            if base != t and not self.catalog.tables[t].per_customer:
                raise CannotAnswer("ranking by a per-event column is not supported")
            idc = ID_COLUMNS.get(base)
            direction = "DESC" if top.group(1) in ("top", "highest") else "ASC"
            select = [(base, ref(base, idc), idc), (t, ref(t, c), c)] if idc != c else [(t, ref(t, c), c)]
            order = [f"{ref(t, c)} {direction}", ref(base, idc)]
            limit = int(top.group(2))
            group_spans.add(s.start)
        elif common:
            s = col_after(common.end(), 20)
            if not s:
                raise CannotAnswer("could not tell which column to count values of")
            t, c = s.payload
            base = t
            group = [(t, ref(t, c), c)]
            select = [(t, "COUNT(*)", "n")]
            order = ["n DESC", ref(t, c)]
            group_spans.add(s.start)
        else:
            agg = re.search(r"\b(average|avg|mean|total|sum of|sum|median)\b", q)
            count = re.search(r"\b(how many|number of|count of|count)\b", q)
            if any(c.payload == "churn_rate" for c in concepts):
                base = "customers"
                select = [("customers", f"AVG({ref('customers', 'churned')})", "churn_rate")]
            elif agg:
                s = col_after(agg.end(), 25)
                if not s:
                    raise CannotAnswer(f"could not tell what to {agg.group(1)}")
                t, c = s.payload
                base = t
                fn = "SUM" if agg.group(1).startswith(("total", "sum")) else "AVG"
                select = [(t, f"{fn}({ref(t, c)})", f"{fn.lower()}_{c}")]
                group_spans.add(s.start)
            elif count:
                ent = next((x for x in tables if x.start >= count.start()), None) or next(iter(tables), None)
                base = ent.payload if ent else "customers"
                select = [(base, "COUNT(*)", "n")]
            else:
                listing = re.search(r"\b(list|show|which|who)\b", q)
                if not listing:
                    raise CannotAnswer("no metric found (try 'churn rate', 'how many', 'average ...', 'top N ... by ...')")
                base = "customers"
                select = [("customers", ref("customers", "customer_id"), "customer_id")]
                order = [ref("customers", "customer_id")]
                limit = DEFAULT_LIST_LIMIT

        # --- grouping: "by X", "per X", "for each X", "broken down by X", "across X"
        for m in re.finditer(r"\b(?:by|per|for each|each|across)\s+(?:the\s+)?", q):
            if top and m.start() >= top.start() and m.start() < top.end():
                continue
            s = col_after(m.end(), 0)
            if s and s.start not in group_spans:
                t, c = s.payload
                group.append((t, ref(t, c), c))
                group_spans.add(s.start)
        if any(c.payload == "per_month" for c in concepts):
            date_col = DATE_COLUMNS.get(base)
            if not date_col:
                raise CannotAnswer(f"{base} has no date column to group by month")
            expr = ref(base, date_col) if date_col == "month" else f"substr({ref(base, date_col)}, 1, 7)"
            group.append((base, expr, "month"))

        # --- filters
        where: list[tuple[str, str, list]] = []  # (table, fragment, params)
        for s in concepts:
            if s.payload in ("churned", "retained") and not any(c.payload == "churn_rate" for c in concepts):
                where.append(("customers", f"{ref('customers', 'churned')} = ?", [1 if s.payload == "churned" else 0]))
        values = [s for s in spans if s.kind == "value"]
        by_pos: dict[int, list[Span]] = {}
        for s in values:
            by_pos.setdefault(s.start, []).append(s)
        involved = {base} | {t for t, _, _ in select + group}
        for pos, cands in sorted(by_pos.items()):
            end = cands[0].end
            near = [c for c in columns if 0 <= c.start - end <= 1 or 0 <= pos - c.end <= 1]
            pick = None
            for cand in cands:
                t, c, v = cand.payload
                if any(n.payload == (t, c) for n in near):
                    pick = cand
                    break
            if pick is None and len(cands) == 1:
                pick = cands[0]
            if pick is None:
                pick = next((c for c in cands if c.payload[0] in involved), None)
            if pick is None:
                continue
            t, c, v = pick.payload
            if any(g[1] == ref(t, c) for g in group):
                continue  # "by contract" grouping, not a filter
            where.append((t, f"{ref(t, c)} = ?", [v]))
            for n in near:
                if n.payload == (t, c):
                    group_spans.add(n.start)
            used_values.add(pos)

        # "top churn drivers" = rank 1
        for s in columns:
            if s.payload == ("churn_drivers", "feature") and re.search(r"\b(top|main|biggest|primary|leading)\s+$", q[: s.start]):
                where.append(("churn_drivers", f"{ref('churn_drivers', 'rank')} = ?", [1]))

        # numeric comparison right after a column: "tenure under 6", "monthly charges above 900"
        for s in columns:
            m = re.match(rf"\s+(?:of\s+|is\s+)?({CMP_RE})\s+(\d+(?:\.\d+)?)", q[s.end:])
            if m:
                t, c = s.payload
                num = float(m.group(2)) if "." in m.group(2) else int(m.group(2))
                where.append((t, f"{ref(t, c)} {CMP_OP[m.group(1)]} ?", [num]))

        # child-row counts: "customers with more than 3 support tickets"
        for m in re.finditer(rf"\b(?:with|having|who had|that had)\s+({CMP_RE})\s+(\d+)\s+", q):
            child = next((x for x in tables if x.start == m.end()), None)
            if child and not self.catalog.tables[child.payload].per_customer and base in ("customers", "churn_scores"):
                ct = child.payload
                where.append((base, f"(SELECT COUNT(*) FROM {ct} WHERE {ct}.customer_id = {ref(base, 'customer_id')}) "
                                    f"{CMP_OP[m.group(1)]} ?", [int(m.group(2))]))

        # --- joins
        needed = {t for t, _, _ in select + group + where} - {base}
        joins, where_sql = [], []
        for t in sorted(needed):
            info = self.catalog.tables[t]
            if not info.has_customer_id or not self.catalog.tables[base].has_customer_id:
                raise CannotAnswer(f"cannot relate {t} to {base}")
            if info.per_customer:
                joins.append(f"JOIN {t} {self.alias[t]} ON {self.alias[t]}.customer_id = {ref(base, 'customer_id')}")
            elif any(tt == t for tt, _, _ in select + group):
                raise CannotAnswer(f"grouping {base} by a per-row column of {t} would double count")
        for t, frag, params in where:
            if t in needed and not self.catalog.tables[t].per_customer:
                where_sql.append((f"{ref(base, 'customer_id')} IN (SELECT customer_id FROM {t} {self.alias[t]} WHERE {frag})", params))
            else:
                where_sql.append((frag, params))

        spec = QuerySpec(base=base, select=[(e, a) for _, e, a in group] + [(e, a) for _, e, a in select],
                         joins=joins, where=where_sql, group_by=[e for _, e, _ in group],
                         order_by=order or [e for _, e, _ in group], limit=limit)
        return render(spec, self.alias)


def render(spec: QuerySpec, alias: dict[str, str]) -> tuple[str, list]:
    sql = "SELECT " + ", ".join(f"{e} AS {a}" for e, a in spec.select)
    sql += f" FROM {spec.base} {alias[spec.base]}"
    for j in spec.joins:
        sql += f" {j}"
    params: list = []
    if spec.where:
        sql += " WHERE " + " AND ".join(f for f, _ in spec.where)
        for _, p in spec.where:
            params += p
    if spec.group_by:
        sql += " GROUP BY " + ", ".join(spec.group_by)
    if spec.order_by:
        sql += " ORDER BY " + ", ".join(spec.order_by)
    if spec.limit is not None:
        sql += " LIMIT ?"
        params.append(spec.limit)
    return sql, params


# =========================================================================== copilot


@dataclass
class Answer:
    question: str
    sql: Optional[str] = None
    params: list = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    source: Optional[str] = None  # "llm:<provider>" or "rules"
    attempts: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self, max_rows: int = 10) -> str:
        if self.error:
            return f"Could not answer: {self.error}"
        if not self.rows:
            return "No matching rows."
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        lines = [", ".join(f"{c}={fmt(v)}" for c, v in zip(self.columns, r)) for r in self.rows[:max_rows]]
        more = f"\n... {len(self.rows) - max_rows} more rows" if len(self.rows) > max_rows else ""
        return "\n".join(lines) + more

    def as_dict(self) -> dict:
        return {"question": self.question, "sql": self.sql, "params": self.params, "columns": self.columns,
                "rows": [list(r) for r in self.rows], "truncated": self.truncated, "source": self.source,
                "attempts": self.attempts, "error": self.error, "latency_ms": round(self.latency_ms, 3)}


class SQLCopilot:
    def __init__(self, db_path: Path = DB_PATH, provider: Optional[LLMProvider] = None):
        self.catalog = SchemaCatalog.introspect(db_path)
        self.db = ReadOnlyDatabase(db_path, allowed_tables=list(self.catalog.tables))
        self.planner = RulePlanner(self.catalog)
        self.provider = provider

    def _run(self, answer: Answer, source: str, sql: str, params: list) -> bool:
        attempt = {"source": source, "sql": sql, "params": params}
        answer.attempts.append(attempt)
        check = validate_sql(sql)
        if not check.ok:
            attempt["error"] = f"rejected by validator: {check.reason}"
            return False
        if count_placeholders(sql) != len(params):
            attempt["error"] = f"{count_placeholders(sql)} placeholders but {len(params)} params"
            return False
        try:
            result: QueryResult = self.db.execute(sql, params)
        except (UnsafeSQLError, TimeoutError, sqlite3.Error) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            return False
        answer.sql, answer.params, answer.source = sql, params, source
        answer.columns, answer.rows, answer.truncated = result.columns, result.rows, result.truncated
        return True

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        answer = Answer(question)
        if self.provider is not None:
            try:
                sql, params = _parse_llm_json(self.provider.complete(
                    SYSTEM_PROMPT, f"SCHEMA:\n{self.catalog.prompt()}\n\nQUESTION: {question}"))
                if self._run(answer, f"llm:{self.provider.name}", sql, params):
                    answer.latency_ms = (time.perf_counter() - start) * 1000
                    return answer
            except (LLMError, ValueError) as exc:
                answer.attempts.append({"source": f"llm:{self.provider.name}", "error": str(exc)})
        try:
            sql, params = self.planner.plan(question)
            if not self._run(answer, "rules", sql, params):
                answer.error = answer.attempts[-1]["error"]
        except CannotAnswer as exc:
            answer.attempts.append({"source": "rules", "error": str(exc)})
            answer.error = str(exc)
        answer.latency_ms = (time.perf_counter() - start) * 1000
        return answer


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the churn database a question")
    ap.add_argument("question")
    ap.add_argument("--provider", help="gemini | openai | groq | none (default: from environment)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ)
    if args.provider:
        env["CHURNINTEL_LLM"] = args.provider
    copilot = SQLCopilot(provider=provider_from_env(env))
    answer = copilot.ask(args.question)
    if args.json:
        print(json.dumps(answer.as_dict(), indent=2, default=str))
        return
    print(f"-- source: {answer.source}\n{answer.sql}\n-- params: {answer.params}\n")
    print(answer.summary())


if __name__ == "__main__":
    main()
