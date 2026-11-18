"""Build data/churn.db: a synthetic subscription business observed at a snapshot date.

    python -m src.build_db            # writes data/churn.db and data/oracle_churn_prob.csv

Timeline:
    signup ... 2025-01 .. 2025-12 (usage, tickets, payments) | SNAPSHOT 2025-12-31 | churn window 2026-01-01 .. 2026-03-31

* Every behavioural table (usage_monthly, support_tickets, payments) only holds events up to the snapshot.
* `customers.churned` / `customers.churn_date` record whether the customer cancelled in the 90 days *after*
  the snapshot. They are outcome columns: the copilot may report on them, the churn model may not use them
  as features (`churn_model.OUTCOME_COLUMNS`).
* Churn is drawn from a hidden logistic process that depends on observable behaviour *and* an unobserved
  disengagement factor, so no model can reach perfect AUC. The generator's own probability is written to
  data/oracle_churn_prob.csv (outside the database, so neither the copilot nor the model can read it) and is
  reported as the ceiling.

All data is synthetic. Amounts are in INR.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "churn.db"
ORACLE_PATH = ROOT / "data" / "oracle_churn_prob.csv"
SNAPSHOT = date(2025, 12, 31)
HORIZON_DAYS = 90
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]
SEED = 2025
N_CUSTOMERS = 6000

REGIONS = ("North", "South", "East", "West", "Central")
SEGMENTS = ("Consumer", "SMB")
CONTRACTS = ("Month-to-Month", "One-Year", "Two-Year")
PLANS = ("Basic", "Standard", "Premium")
PAYMENT_METHODS = ("Credit card", "Bank transfer", "Electronic check", "UPI")
TICKET_CATEGORIES = ("Billing", "Technical", "Network", "Account")
SEVERITIES = ("low", "medium", "high")

DATA_DICTIONARY = {
    "customers": {
        "customer_id": "Customer identifier",
        "signup_date": "Date the subscription started (YYYY-MM-DD)",
        "region": "Service region: North, South, East, West, Central",
        "segment": "Consumer or SMB",
        "age": "Account holder age in years",
        "contract": "Contract type: Month-to-Month, One-Year, Two-Year",
        "plan": "Plan tier: Basic, Standard, Premium",
        "payment_method": "Credit card, Bank transfer, Electronic check, UPI",
        "paperless_billing": "1 if billed electronically",
        "monthly_charges": "Current monthly charge (INR)",
        "tenure_months": "Months since signup at the snapshot date (2025-12-31)",
        "churned": "OUTCOME: 1 if the customer cancelled between 2026-01-01 and 2026-03-31",
        "churn_date": "OUTCOME: cancellation date, NULL if not churned",
    },
    "usage_monthly": {
        "customer_id": "Customer identifier",
        "month": "Calendar month YYYY-MM (2025-01 .. 2025-12)",
        "data_gb": "Mobile data used in the month (GB)",
        "call_minutes": "Voice minutes in the month",
        "outage_minutes": "Network outage minutes experienced in the month",
    },
    "support_tickets": {
        "ticket_id": "Ticket identifier",
        "customer_id": "Customer identifier",
        "opened_at": "Date opened (YYYY-MM-DD)",
        "category": "Billing, Technical, Network, Account",
        "severity": "low, medium, high",
        "resolved_hours": "Hours to resolution",
        "csat": "Satisfaction score 1-5 left after resolution; NULL if no survey response",
    },
    "payments": {
        "payment_id": "Payment identifier",
        "customer_id": "Customer identifier",
        "due_date": "Bill due date (YYYY-MM-DD)",
        "amount": "Billed amount (INR)",
        "status": "paid, late, failed",
    },
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = N_CUSTOMERS, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)

    tenure = rng.integers(3, 85, n)
    signup = [SNAPSHOT - timedelta(days=int(t * 30.44) + int(d)) for t, d in zip(tenure, rng.integers(0, 28, n))]
    region = rng.choice(len(REGIONS), n, p=[0.24, 0.22, 0.20, 0.20, 0.14])
    segment = rng.choice(2, n, p=[0.8, 0.2])
    age = np.clip(np.round(rng.normal(41, 13, n)), 18, 80).astype(int)
    m2m_p = np.clip(0.70 - 0.006 * tenure, 0.25, 0.70)
    u = rng.random(n)
    contract = np.where(u < m2m_p, 0, np.where(u < m2m_p + (1 - m2m_p) * 0.55, 1, 2))
    plan = rng.choice(3, n, p=[0.4, 0.4, 0.2])
    payment = rng.choice(4, n, p=[0.30, 0.25, 0.25, 0.20])
    paperless = (rng.random(n) < 0.6).astype(int)
    base_price = np.array([399.0, 699.0, 1099.0])[plan]
    charges = np.round(base_price * np.array([1.0, 0.95, 0.90])[contract] * rng.normal(1.0, 0.08, n) + 100 * segment, 2)

    latent = rng.standard_normal(n)  # unobserved disengagement
    region_outage = np.array([12.0, 20.0, 9.0, 14.0, 30.0])[region]

    # ---- monthly usage (12 months)
    plan_gb = np.array([6.0, 14.0, 30.0])[plan]
    habit = rng.lognormal(0, 0.35, n)
    decliners = latent > 0.6
    usage_rows = []
    outage = np.zeros((n, 12))
    for m in range(12):
        decay = np.where(decliners & (m >= 9), 1 - 0.18 * (m - 8) * np.clip(latent, 0, 2), 1.0)
        gb = plan_gb * habit * rng.lognormal(0, 0.15, n) * np.clip(decay, 0.2, 1)
        minutes = np.round(rng.gamma(4, 60, n) * np.clip(decay, 0.3, 1))
        spike = rng.random(n) < 0.04
        outage[:, m] = rng.poisson(region_outage) + spike * rng.integers(60, 400, n)
        usage_rows.append(pd.DataFrame({"customer_id": ids, "month": MONTHS[m], "data_gb": np.round(gb, 2),
                                        "call_minutes": minutes.astype(int), "outage_minutes": outage[:, m].astype(int)}))
    usage = pd.concat(usage_rows, ignore_index=True)

    # ---- support tickets (Poisson per month; outages and disengagement drive them)
    tickets = []
    last90 = np.zeros(n)
    high90 = np.zeros(n)
    bad_csat = np.zeros(n)
    long_wait = np.zeros(n)
    for m in range(12):
        lam = 0.07 * np.exp(0.35 * latent + 0.004 * outage[:, m] + 0.2 * (contract == 0))
        k = rng.poisson(lam)
        for cid_idx in np.flatnonzero(k):
            for _ in range(k[cid_idx]):
                day = int(rng.integers(1, 29))
                cat = rng.choice(4, p=[0.30, 0.30, 0.25, 0.15]) if outage[cid_idx, m] < 100 else 2
                sev = rng.choice(3, p=[0.5, 0.35, 0.15]) if cat != 2 else rng.choice(3, p=[0.3, 0.4, 0.3])
                hours = float(np.round(rng.lognormal(np.log([6, 18, 36][sev]), 0.6), 1))
                csat = None if rng.random() < 0.35 else int(np.clip(np.round(5 - hours / 20 + rng.normal(0, 0.8)), 1, 5))
                tickets.append((int(ids[cid_idx]), f"{MONTHS[m]}-{day:02d}", TICKET_CATEGORIES[cat], SEVERITIES[sev], hours, csat))
                if m >= 9:
                    last90[cid_idx] += 1
                    high90[cid_idx] += sev == 2
                    long_wait[cid_idx] += hours > 48
                if csat is not None and csat <= 2:
                    bad_csat[cid_idx] = 1
    tickets = pd.DataFrame(tickets, columns=["customer_id", "opened_at", "category", "severity", "resolved_hours", "csat"])
    tickets = tickets.sort_values(["opened_at", "customer_id"], kind="stable").reset_index(drop=True)
    tickets.insert(0, "ticket_id", np.arange(1, len(tickets) + 1))

    # ---- payments (one bill per month)
    pay_rows = []
    failed6 = np.zeros(n)
    fail_base = np.array([0.01, 0.01, 0.05, 0.02])[payment]
    for m in range(12):
        p_fail = np.clip(fail_base * np.exp(0.5 * latent), 0, 0.5)
        p_late = np.clip(0.04 * np.exp(0.3 * latent), 0, 0.4)
        r = rng.random(n)
        status = np.where(r < p_fail, "failed", np.where(r < p_fail + p_late, "late", "paid"))
        if m >= 6:
            failed6 += status == "failed"
        pay_rows.append(pd.DataFrame({"customer_id": ids, "due_date": f"{MONTHS[m]}-05", "amount": charges, "status": status}))
    payments = pd.concat(pay_rows, ignore_index=True)
    payments.insert(0, "payment_id", np.arange(1, len(payments) + 1))

    # ---- churn in the 90 days after the snapshot (hidden process; fixed before any model was trained)
    recent_gb = np.stack([usage.loc[usage["month"] == mo, "data_gb"].to_numpy() for mo in MONTHS[9:]]).mean(axis=0)
    prior_gb = np.stack([usage.loc[usage["month"] == mo, "data_gb"].to_numpy() for mo in MONTHS[3:9]]).mean(axis=0)
    usage_drop = np.clip(1 - recent_gb / prior_gb, 0, 1)
    logit = (
        -2.9
        + np.array([1.0, 0.0, -1.2])[contract]
        - 0.45 * np.log1p(tenure / 12)
        + 0.0012 * (charges - base_price)
        + 0.30 * last90 + 0.45 * high90 + 0.35 * long_wait + 0.45 * bad_csat
        + 0.003 * outage[:, 9:].sum(axis=1) / 3
        + 2.2 * usage_drop
        + 0.55 * failed6
        + 0.25 * (payment == 2)
        - 0.15 * paperless
        + 0.20 * (segment == 0)
        + 0.55 * latent
    )
    p_churn = _sigmoid(logit)
    churned = (rng.random(n) < p_churn).astype(int)
    churn_date = [str(SNAPSHOT + timedelta(days=int(d))) if c else None
                  for c, d in zip(churned, rng.integers(1, HORIZON_DAYS + 1, n))]

    customers = pd.DataFrame({
        "customer_id": ids, "signup_date": [str(s) for s in signup], "region": np.array(REGIONS)[region],
        "segment": np.array(SEGMENTS)[segment], "age": age, "contract": np.array(CONTRACTS)[contract],
        "plan": np.array(PLANS)[plan], "payment_method": np.array(PAYMENT_METHODS)[payment],
        "paperless_billing": paperless, "monthly_charges": charges, "tenure_months": tenure,
        "churned": churned, "churn_date": churn_date,
    })
    oracle = pd.DataFrame({"customer_id": ids, "oracle_churn_prob": np.round(p_churn, 6)})
    return {"customers": customers, "usage_monthly": usage, "support_tickets": tickets, "payments": payments, "_oracle": oracle}


DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, signup_date TEXT NOT NULL, region TEXT NOT NULL, segment TEXT NOT NULL,
    age INTEGER NOT NULL, contract TEXT NOT NULL, plan TEXT NOT NULL, payment_method TEXT NOT NULL,
    paperless_billing INTEGER NOT NULL, monthly_charges REAL NOT NULL, tenure_months INTEGER NOT NULL,
    churned INTEGER NOT NULL, churn_date TEXT
);
CREATE TABLE usage_monthly (
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id), month TEXT NOT NULL, data_gb REAL NOT NULL,
    call_minutes INTEGER NOT NULL, outage_minutes INTEGER NOT NULL, PRIMARY KEY (customer_id, month)
);
CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    opened_at TEXT NOT NULL, category TEXT NOT NULL, severity TEXT NOT NULL, resolved_hours REAL NOT NULL, csat INTEGER
);
CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    due_date TEXT NOT NULL, amount REAL NOT NULL, status TEXT NOT NULL
);
CREATE TABLE data_dictionary (table_name TEXT NOT NULL, column_name TEXT NOT NULL, description TEXT NOT NULL,
    PRIMARY KEY (table_name, column_name));
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_payments_customer ON payments(customer_id);
"""


def build(db_path: Path = DB_PATH, oracle_path: Path = ORACLE_PATH, n: int = N_CUSTOMERS, seed: int = SEED) -> Path:
    tables = generate(n, seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(DDL)
        for name in ("customers", "usage_monthly", "support_tickets", "payments"):
            df = tables[name]
            rows = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
            con.executemany(f"INSERT INTO {name} ({', '.join(df.columns)}) VALUES ({', '.join('?' * len(df.columns))})", rows)
        con.executemany("INSERT INTO data_dictionary VALUES (?, ?, ?)",
                        [(t, c, d) for t, cols in DATA_DICTIONARY.items() for c, d in cols.items()])
        con.commit()
    finally:
        con.close()
    tables["_oracle"].to_csv(oracle_path, index=False)
    return db_path


def table_fingerprint(db_path: Path, tables=("customers", "usage_monthly", "support_tickets", "payments")) -> str:
    """Content hash of the source tables (independent of SQLite page layout)."""
    h = hashlib.sha256()
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        for t in tables:
            for row in con.execute(f"SELECT * FROM {t} ORDER BY 1, 2"):
                h.update(repr(row).encode())
    finally:
        con.close()
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the synthetic churn database")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    path = build(Path(args.db))
    con = sqlite3.connect(path)
    for t in ("customers", "usage_monthly", "support_tickets", "payments"):
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    print("churn rate", round(con.execute("SELECT AVG(churned) FROM customers").fetchone()[0], 4))
    con.close()
    print("fingerprint", table_fingerprint(path))
