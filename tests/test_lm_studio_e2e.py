import os

import pytest

from src.agents.portfolio_manager import generate_trading_decision
from src.main import run_hedge_fund


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LM_STUDIO_E2E") != "1",
    reason="Set RUN_LM_STUDIO_E2E=1 to test against a running LM Studio server",
)


def test_lm_studio_returns_a_parsed_portfolio_decision(monkeypatch):
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen3.5-122b-a10b/ud")
    monkeypatch.setenv("OPENAI_API_BASE", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")

    state = {
        "metadata": {
            "model_name": model_name,
            "model_provider": "OpenAI",
        }
    }

    result = generate_trading_decision(
        tickers=["AAPL"],
        signals_by_ticker={
            "AAPL": {
                "technical_analyst_agent": {
                    "sig": "bullish",
                    "conf": 80,
                }
            }
        },
        current_prices={"AAPL": 100.0},
        max_shares={"AAPL": 10},
        portfolio={
            "cash": 1_000.0,
            "equity": 1_000.0,
            "margin_requirement": 0.5,
            "margin_used": 0.0,
            "positions": {},
        },
        agent_id="portfolio_manager",
        state=state,
    )

    assert set(result.decisions) == {"AAPL"}
    assert result.decisions["AAPL"].reasoning != "Default decision: hold"


def test_full_aapl_workflow_with_yfinance_and_lm_studio(monkeypatch):
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen3.5-122b-a10b/ud")
    monkeypatch.setenv("OPENAI_API_BASE", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    portfolio = {
        "cash": 100_000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            "AAPL": {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
        },
        "realized_gains": {"AAPL": {"long": 0.0, "short": 0.0}},
    }

    result = run_hedge_fund(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        portfolio=portfolio,
        selected_analysts=["technical_analyst"],
        model_name=model_name,
        model_provider="OpenAI",
    )

    assert "technical_analyst_agent" in result["analyst_signals"]
    assert set(result["decisions"]) == {"AAPL"}
    assert result["decisions"]["AAPL"]["reasoning"] != "Default decision: hold"
