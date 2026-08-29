# ⚡ Dynamic Prediction Engine (Nifty & BankNifty Algo Radar)

An institutional-grade algorithmic option trading intelligence and prediction system combining **Upstox API v2 Option Chain Dynamics**, **5-Minute Multi-Indicator Technical Analysis**, and **FinBERT AI News Sentiment Scoring**.

Includes a mobile-first OLED dark mode **Streamlit Dashboard** and a high-performance **FastAPI REST API Backend**.

---

## 🌟 Key Features

- **Upstox API v2 Option Chain Engine**:
  - Live Spot LTP, Strike-wise Call/Put OI, and Change in OI.
  - Automated Put-Call Ratio (PCR) and Intrinsic Max Pain calculation.
  - Smart Money Shift Detection (Call Unwinding, Put Writing, Support/Resistance discovery).
- **AI Financial News Sentiment (FinBERT)**:
  - Real-time RSS scraping from Moneycontrol and The Economic Times via `feedparser`.
  - HuggingFace `ProsusAI/finbert` transformer pipeline scoring headlines into a $[-1.0, +1.0]$ polarity scale.
- **5-Minute Technical Indicators**:
  - Volume Weighted Average Price (VWAP).
  - ATR-based Supertrend line and directional shifts ($+1 / -1$).
  - 9 & 21 Exponential Moving Average (EMA) crossovers and slope momentum.
  - Average Directional Index (ADX) with $+DI / -DI$ trend strength filtering.
- **Institutional Option Buying Strategy**:
  - **`BUY_ATM_CE`**: $\text{PCR} > 1.10$ + Call OI Unwinding + $\text{Spot} > \text{VWAP}$ + $\text{Sentiment} > +0.25$.
  - **`BUY_ATM_PE`**: $\text{PCR} < 0.85$ + Put OI Unwinding + $\text{Spot} < \text{VWAP}$ + $\text{Sentiment} < -0.25$.
  - **`NO_TRADE`**: Range-bound / Simultaneous heavy Call & Put writing (short straddle theta trap).
  - Dynamic **12% Stop-Loss** on premium and **1:2 Risk-to-Reward Target (24% Profit)**.
- **Mobile-First Streamlit Dashboard**:
  - OLED high-contrast dark theme with color-coded prediction cards (Green for CE, Red for PE, Gray for No Trade).
  - Semi-circular live PCR Sentiment Gauge.
  - Strike-wise Call vs Put Change in OI grouped bar charts with ATM marker.
  - Auto-refresh tick timer (15s, 30s, 60s).

---

## 📁 Repository Structure

```
├── app.py                     # Mobile-First Streamlit Dashboard
├── api_server.py              # High-Performance FastAPI REST Server (/api/predict)
├── data/
│   ├── __init__.py            # Package initialization & exports
│   └── option_chain.py        # Upstox v2 Option Chain Engine & WebSocket client
├── sentiment_technical.py     # FinBERT News Sentiment & Technical Analysis Pipeline
├── test_app.py                # Automated Test Suite for Trade Predictions & Risk Logic
├── requirements.txt           # Python dependencies
└── .gitignore                 # Standard Python gitignore
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/swappy2122/Nifty-call.git
cd Nifty-call
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard (UI)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your mobile browser or desktop.

### 4. Run the FastAPI Backend Server (REST API)
```bash
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Run Automated Tests
```bash
python3 test_app.py
```

---

## 📊 API Reference

### `POST /api/predict`
Calculates dynamic option buying trade recommendations with complete risk metrics.

#### Request Body Example:
```json
{
  "symbol": "NIFTY",
  "spot_price": 24550.0,
  "vwap": 24500.0,
  "pcr": 1.25,
  "sentiment_score": 0.35,
  "atm_strike": 24550.0,
  "atm_ce_ltp": 150.0,
  "atm_pe_ltp": 120.0,
  "call_change_oi": -30000,
  "put_change_oi": 50000
}
```

#### Response Example:
```json
{
  "timestamp": "2026-08-29T16:01:40Z",
  "symbol": "NIFTY",
  "signal": "BUY_ATM_CE",
  "confidence": 0.69,
  "trade_setup": {
    "action": "BUY_ATM_CE",
    "strike": 24550.0,
    "option_type": "CE",
    "entry_price": 150.0,
    "stop_loss": 132.0,
    "target": 186.0,
    "risk_percentage": 12.0,
    "reward_percentage": 24.0,
    "risk_reward_ratio": "1:2",
    "potential_risk_points": 18.0,
    "potential_reward_points": 36.0
  },
  "conditions_met": {
    "pcr_condition_met": true,
    "pcr_value": 1.25,
    "pcr_requirement": "PCR > 1.1 for CE, PCR < 0.85 for PE",
    "oi_unwinding_condition_met": true,
    "oi_unwinding_detail": "Call Unwinding: True (Chg OI: -30,000)",
    "vwap_condition_met": true,
    "spot_vs_vwap_diff": 50.0,
    "sentiment_condition_met": true,
    "sentiment_score": 0.35,
    "sentiment_requirement": "Sentiment > +0.25 for CE, < -0.25 for PE"
  },
  "reason": "BUY ATM CE triggered: Bullish confluence confirmed. PCR (1.25 > 1.1), Call OI Unwinding observed, Spot > VWAP, and News Sentiment > +0.25."
}
```

---

## 🛡️ License
MIT License. Created for algorithmic options intelligence and dynamic market prediction.
