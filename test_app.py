"""
Test Suite for FastAPI Dynamic Prediction Engine
================================================
Verifies all signal logic, risk/reward calculations, and JSON outputs.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    print("✓ Health check passed.")


def test_buy_atm_ce_signal():
    payload = {
        "symbol": "NIFTY",
        "spot_price": 24550.0,
        "vwap": 24500.0,  # Price > VWAP
        "pcr": 1.25,      # PCR > 1.1
        "sentiment_score": 0.35,  # Sentiment > 0.25
        "atm_strike": 24550.0,
        "atm_ce_ltp": 150.0,
        "atm_pe_ltp": 120.0,
        "call_change_oi": -30000,  # Call OI Unwinding
        "put_change_oi": 50000,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["signal"] == "BUY_ATM_CE"
    trade = data["trade_setup"]
    assert trade["action"] == "BUY_ATM_CE"
    assert trade["strike"] == 24550.0
    assert trade["option_type"] == "CE"
    assert trade["entry_price"] == 150.0

    # 12% Stop Loss: 150 * 0.88 = 132.0
    assert trade["stop_loss"] == 132.0
    # 1:2 R:R Target (24% profit): 150 * 1.24 = 186.0
    assert trade["target"] == 186.0
    assert trade["risk_percentage"] == 12.0
    assert trade["reward_percentage"] == 24.0
    assert trade["risk_reward_ratio"] == "1:2"
    print("✓ BUY_ATM_CE test passed with 12% SL and 1:2 Target.")


def test_buy_atm_pe_signal():
    payload = {
        "symbol": "BANKNIFTY",
        "spot_price": 51200.0,
        "vwap": 51350.0,  # Price < VWAP
        "pcr": 0.72,      # PCR < 0.85
        "sentiment_score": -0.40,  # Sentiment < -0.25
        "atm_strike": 51200.0,
        "atm_ce_ltp": 200.0,
        "atm_pe_ltp": 250.0,
        "call_change_oi": 40000,
        "put_change_oi": -25000,  # Put OI Unwinding
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["signal"] == "BUY_ATM_PE"
    trade = data["trade_setup"]
    assert trade["action"] == "BUY_ATM_PE"
    assert trade["strike"] == 51200.0
    assert trade["option_type"] == "PE"
    assert trade["entry_price"] == 250.0

    # 12% Stop Loss: 250 * 0.88 = 220.0
    assert trade["stop_loss"] == 220.0
    # 1:2 R:R Target: 250 * 1.24 = 310.0
    assert trade["target"] == 310.0
    print("✓ BUY_ATM_PE test passed with 12% SL and 1:2 Target.")


def test_no_trade_rangebound():
    payload = {
        "symbol": "NIFTY",
        "spot_price": 24500.0,
        "vwap": 24500.0,
        "pcr": 0.98,  # In neutral range [0.85, 1.10]
        "sentiment_score": 0.05,  # In neutral range
        "atm_strike": 24500.0,
        "atm_ce_ltp": 130.0,
        "atm_pe_ltp": 130.0,
        "call_change_oi": 15000,
        "put_change_oi": 15000,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["signal"] == "NO_TRADE"
    trade = data["trade_setup"]
    assert trade["action"] == "NO_TRADE"
    assert trade["entry_price"] == 0.0
    assert trade["stop_loss"] == 0.0
    assert trade["target"] == 0.0
    print("✓ NO_TRADE test passed.")


if __name__ == "__main__":
    print("Running FastAPI Prediction Endpoint Tests...")
    test_health()
    test_buy_atm_ce_signal()
    test_buy_atm_pe_signal()
    test_no_trade_rangebound()
    print("All tests passed successfully!")
