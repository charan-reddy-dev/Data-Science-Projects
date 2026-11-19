import json
import sqlite3

import httpx
import pytest

from src.sql_copilot import (GeminiProvider, OpenAICompatibleProvider, RulePlanner, SchemaCatalog, SQLCopilot,
                             provider_from_env)


class ScriptedProvider:
    name = "scripted"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.reply


def test_schema_is_introspected_from_the_database(small_db):
    cat = SchemaCatalog.introspect(small_db)
    assert {"customers", "support_tickets", "payments", "usage_monthly", "churn_scores", "churn_drivers"} <= set(cat.tables)
    assert cat.tables["customers"].columns["contract"].values == ["Month-to-Month", "One-Year", "Two-Year"]
    assert cat.tables["customers"].per_customer and not cat.tables["support_tickets"].per_customer
    assert ("customer_id", "customers", "customer_id") in cat.tables["payments"].foreign_keys
    prompt = cat.prompt()
    assert "'Electronic check'" in prompt and "OUTCOME" in prompt


@pytest.mark.parametrize("question,expected_sql", [
    ("What is the churn rate by contract type?", "SELECT contract, AVG(churned) FROM customers GROUP BY contract"),
    ("How many customers with tenure under 6 months churned?",
     "SELECT COUNT(*) FROM customers WHERE tenure_months < 6 AND churned = 1"),
    ("Number of support tickets by category", "SELECT category, COUNT(*) FROM support_tickets GROUP BY category"),
    ("How many high risk customers are there by contract type?",
     "SELECT c.contract, COUNT(*) FROM churn_scores s JOIN customers c USING (customer_id) WHERE s.risk_tier = 'High' GROUP BY 1"),
])
def test_planner_matches_reference_queries(small_db, question, expected_sql):
    cp = SQLCopilot(small_db)
    a = cp.ask(question)
    con = sqlite3.connect(small_db)
    assert a.ok and a.source == "rules"
    assert sorted(a.rows) == sorted(con.execute(expected_sql).fetchall())
    assert "'" not in a.sql  # literals are bound parameters
    con.close()


def test_planner_refuses_write_requests_and_unknown_questions(small_db):
    cp = SQLCopilot(small_db)
    assert "read-only" in cp.ask("Please delete every churned customer").error
    assert not cp.ask("What's the weather like?").ok


def test_llm_answer_is_used_when_valid(small_db):
    provider = ScriptedProvider(json.dumps({"sql": "SELECT COUNT(*) FROM customers WHERE plan = ?", "params": ["Basic"]}))
    cp = SQLCopilot(small_db, provider=provider)
    a = cp.ask("How many Basic customers?")
    assert a.source == "llm:scripted" and a.params == ["Basic"] and a.rows[0][0] > 0
    assert "TABLE customers" in provider.prompts[0]  # schema-grounded prompt


@pytest.mark.parametrize("reply", [
    json.dumps({"sql": "DROP TABLE customers", "params": []}),
    json.dumps({"sql": "SELECT 1; DELETE FROM customers", "params": []}),
    json.dumps({"sql": "SELECT * FROM customers WHERE plan = ?", "params": []}),
    json.dumps({"sql": "SELECT nope FROM nowhere", "params": []}),
    "not json at all",
    json.dumps({"sql": "", "refusal": "cannot answer"}),
])
def test_bad_llm_output_falls_back_to_the_planner(small_db, reply):
    cp = SQLCopilot(small_db, provider=ScriptedProvider(reply))
    a = cp.ask("What is the overall churn rate?")
    assert a.ok and a.source == "rules"
    assert a.attempts[0]["source"] == "llm:scripted" and a.attempts[0].get("error")


def test_openai_compatible_adapter():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"sql": "SELECT 1", "params": []}'}}]})

    p = OpenAICompatibleProvider("k", "llama", "https://api.groq.com/openai/v1", name="groq",
                                 client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "SELECT 1" in p.complete("sys", "user")
    assert seen["url"].endswith("/openai/v1/chat/completions") and seen["auth"] == "Bearer k"
    assert seen["body"]["response_format"] == {"type": "json_object"} and seen["body"]["temperature"] == 0


def test_gemini_adapter():
    seen = {}

    def handler(request):
        seen["url"], seen["key"] = str(request.url), request.headers["x-goog-api-key"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"sql": "SELECT 2"}'}]}}]})

    p = GeminiProvider("secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "SELECT 2" in p.complete("sys", "user")
    assert seen["url"].endswith("/v1beta/models/gemini-2.5-flash:generateContent") and seen["key"] == "secret"


def test_provider_selection_from_environment():
    assert provider_from_env({}) is None
    assert provider_from_env({"GEMINI_API_KEY": "g"}).name == "gemini"
    assert provider_from_env({"GROQ_API_KEY": "q"}).name == "groq"
    assert provider_from_env({"GEMINI_API_KEY": "g", "CHURNINTEL_LLM": "none"}) is None
    with pytest.raises(ValueError):
        provider_from_env({"CHURNINTEL_LLM": "openai"})
