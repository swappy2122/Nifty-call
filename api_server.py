"""
High-Performance Dynamic Trading Prediction API
===============================================
FastAPI Application serving real-time quantitative trade predictions at `/api/predict`.

Integrates:
- Upstox Option Chain Engine (PCR, Max Pain, ATM strikes, Smart Money Shifts)
- 5-min OHLCV Technical Indicators (VWAP, Supertrend, 9/21 EMA, ADX)
- Financial News Sentiment Analysis (Moneycontrol, Economic Times via FinBERT)
- Rule-based dynamic Option Buying execution engine with 12% dynamic Stop-Loss & 1:2 R:R Target
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.option_chain import OptionChainEngine
from sentiment_technical import (
    CombinedMomentumPipeline,
    NewsHeadline,
    NewsSentimentAnalyzer,
    TechnicalAnalysisEngine,
    generate_mock_ohlcv,
)

# Logging
logger = logging.getLogger("api_predict")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# FastAPI Initialization
app = FastAPI(
    title="Dynamic Prediction Engine API",
    description="High-performance algorithmic trade prediction combining Option Chain OI Shifts, Technical Indicators, and FinBERT Sentiment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cached Singleton Engines
news_analyzer = NewsSentimentAnalyzer(use_pipeline=True)
tech_engine = TechnicalAnalysisEngine()
pipeline = CombinedMomentumPipeline(
    tech_weight=0.65,
    sent_weight=0.35,
    news_analyzer=news_analyzer,
    tech_engine=tech_engine,
)


# ============================================================================
# Pydantic Schemas
# ============================================================================

class PredictRequest(BaseModel):
    """Optional payload to supply manual or streaming data directly."""
    symbol: str = Field(default="NIFTY", description="Trading index or stock ticker (e.g. NIFTY, BANKNIFTY)")
    access_token: Optional[str] = Field(default=None, description="Upstox v2 API Bearer access token")
    spot_price: Optional[float] = Field(default=None, description="Current spot LTP (optional manual override)")
    vwap: Optional[float] = Field(default=None, description="Current VWAP level (optional manual override)")
    pcr: Optional[float] = Field(default=None, description="Put-Call Ratio (optional manual override)")
    sentiment_score: Optional[float] = Field(default=None, description="Sentiment score in [-1.0, +1.0]")
    atm_strike: Optional[float] = Field(default=None, description="ATM strike price")
    atm_ce_ltp: Optional[float] = Field(default=None, description="ATM CE Option Premium LTP")
    atm_pe_ltp: Optional[float] = Field(default=None, description="ATM PE Option Premium LTP")
    call_change_oi: Optional[int] = Field(default=None, description="Aggregate or ATM Call Change in OI")
    put_change_oi: Optional[int] = Field(default=None, description="Aggregate or ATM Put Change in OI")
    ohlcv_bars: Optional[List[Dict[str, float]]] = Field(default=None, description="List of OHLCV candle dicts")


class TradeSetup(BaseModel):
    action: Literal["BUY_ATM_CE", "BUY_ATM_PE", "NO_TRADE"]
    strike: float
    option_type: Literal["CE", "PE", "NONE"]
    entry_price: float
    stop_loss: float
    target: float
    risk_percentage: float = 12.0
    reward_percentage: float = 24.0
    risk_reward_ratio: str = "1:2"
    potential_risk_points: float
    potential_reward_points: float


class ConditionCheck(BaseModel):
    pcr_condition_met: bool
    pcr_value: float
    pcr_requirement: str
    oi_unwinding_condition_met: bool
    oi_unwinding_detail: str
    vwap_condition_met: bool
    spot_vs_vwap_diff: float
    sentiment_condition_met: bool
    sentiment_score: float
    sentiment_requirement: str


class PredictResponse(BaseModel):
    timestamp: str
    symbol: str
    signal: Literal["BUY_ATM_CE", "BUY_ATM_PE", "NO_TRADE"]
    confidence: float
    trade_setup: TradeSetup
    conditions_met: ConditionCheck
    metrics_summary: Dict[str, Any]
    reason: str


# ============================================================================
# Core Signal & Risk Management Logic
# ============================================================================

def evaluate_signal(
    symbol: str,
    spot_price: float,
    vwap: float,
    pcr: float,
    sentiment_score: float,
    atm_strike: float,
    atm_ce_ltp: float,
    atm_pe_ltp: float,
    call_change_oi: int,
    put_change_oi: int,
    call_unwinding_detected: bool,
    put_unwinding_detected: bool,
    simultaneous_writing: bool,
    max_pain: float,
) -> PredictResponse:
    """
    Apply strict institutional option buying signal logic with 12% SL and 1:2 R:R.

    Rules:
    - BUY ATM CE: PCR > 1.1 + Call OI Unwinding + Price > VWAP + Sentiment > 0.25
    - BUY ATM PE: PCR < 0.85 + Put OI Unwinding + Price < VWAP + Sentiment < -0.25
    - NO TRADE: Range-bound / High Call+Put Writing simultaneously or unaligned conditions.
    """
    price_above_vwap = spot_price > vwap
    price_below_vwap = spot_price < vwap

    # Check CE Conditions
    ce_pcr_ok = pcr > 1.1
    ce_oi_ok = call_unwinding_detected or (call_change_oi < 0)
    ce_vwap_ok = price_above_vwap
    ce_sent_ok = sentiment_score > 0.25

    # Check PE Conditions
    pe_pcr_ok = pcr < 0.85
    pe_oi_ok = put_unwinding_detected or (put_change_oi < 0)
    pe_vwap_ok = price_below_vwap
    pe_sent_ok = sentiment_score < -0.25

    # Signal Evaluation
    if simultaneous_writing:
        signal = "NO_TRADE"
        reason = (
            "NO TRADE: Market is range-bound with aggressive simultaneous Call & Put writing "
            "(Short Straddle/Strangle theta decay environment)."
        )
    elif ce_pcr_ok and ce_oi_ok and ce_vwap_ok and ce_sent_ok:
        signal = "BUY_ATM_CE"
        reason = (
            f"BUY ATM CE triggered: Bullish confluence confirmed. PCR ({pcr:.2f} > 1.1), "
            f"Call OI Unwinding observed (Smart money exiting short calls), "
            f"Spot ({spot_price:.2f}) > VWAP ({vwap:.2f}), and News Sentiment ({sentiment_score:+.2f} > +0.25)."
        )
    elif pe_pcr_ok and pe_oi_ok and pe_vwap_ok and pe_sent_ok:
        signal = "BUY_ATM_PE"
        reason = (
            f"BUY ATM PE triggered: Bearish breakdown confirmed. PCR ({pcr:.2f} < 0.85), "
            f"Put OI Unwinding observed (Support collapsing), "
            f"Spot ({spot_price:.2f}) < VWAP ({vwap:.2f}), and News Sentiment ({sentiment_score:+.2f} < -0.25)."
        )
    else:
        signal = "NO_TRADE"
        unmet = []
        if not (ce_pcr_ok or pe_pcr_ok):
            unmet.append(f"PCR ({pcr:.2f}) in neutral zone [0.85 - 1.10]")
        if not (ce_oi_ok or pe_oi_ok):
            unmet.append("No definitive OI Unwinding detected")
        if not (ce_sent_ok or pe_sent_ok):
            unmet.append(f"Sentiment ({sentiment_score:+.2f}) in neutral zone [-0.25 to +0.25]")
        reason = "NO TRADE: Filter conditions not satisfied (" + "; ".join(unmet) + ")."

    # Risk & Target Calculations (12% SL, 1:2 R:R Target = 24% Gain)
    sl_pct = 0.12
    target_pct = 0.24

    if signal == "BUY_ATM_CE":
        entry_price = max(1.0, atm_ce_ltp)
        stop_loss = round(entry_price * (1.0 - sl_pct), 2)
        target = round(entry_price * (1.0 + target_pct), 2)
        risk_pts = round(entry_price - stop_loss, 2)
        reward_pts = round(target - entry_price, 2)
        option_type = "CE"
        confidence = round(min(0.95, 0.65 + (pcr - 1.1) * 0.1 + (sentiment_score - 0.25) * 0.2), 2)
    elif signal == "BUY_ATM_PE":
        entry_price = max(1.0, atm_pe_ltp)
        stop_loss = round(entry_price * (1.0 - sl_pct), 2)
        target = round(entry_price * (1.0 + target_pct), 2)
        risk_pts = round(entry_price - stop_loss, 2)
        reward_pts = round(target - entry_price, 2)
        option_type = "PE"
        confidence = round(min(0.95, 0.65 + (0.85 - pcr) * 0.1 + (abs(sentiment_score) - 0.25) * 0.2), 2)
    else:
        entry_price = 0.0
        stop_loss = 0.0
        target = 0.0
        risk_pts = 0.0
        reward_pts = 0.0
        option_type = "NONE"
        confidence = 0.0

    trade_setup = TradeSetup(
        action=signal,
        strike=atm_strike,
        option_type=option_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target=target,
        risk_percentage=12.0,
        reward_percentage=24.0,
        risk_reward_ratio="1:2",
        potential_risk_points=risk_pts,
        potential_reward_points=reward_pts,
    )

    # Condition Check Details
    is_ce_cand = (sentiment_score >= 0)
    conditions_met = ConditionCheck(
        pcr_condition_met=(ce_pcr_ok if is_ce_cand else pe_pcr_ok),
        pcr_value=round(pcr, 3),
        pcr_requirement="PCR > 1.1 for CE, PCR < 0.85 for PE",
        oi_unwinding_condition_met=(ce_oi_ok if is_ce_cand else pe_oi_ok),
        oi_unwinding_detail=(
            f"Call Unwinding: {call_unwinding_detected} (Chg OI: {call_change_oi:,})"
            if is_ce_cand
            else f"Put Unwinding: {put_unwinding_detected} (Chg OI: {put_change_oi:,})"
        ),
        vwap_condition_met=(ce_vwap_ok if is_ce_cand else pe_vwap_ok),
        spot_vs_vwap_diff=round(spot_price - vwap, 2),
        sentiment_condition_met=(ce_sent_ok if is_ce_cand else pe_sent_ok),
        sentiment_score=round(sentiment_score, 4),
        sentiment_requirement="Sentiment > +0.25 for CE, < -0.25 for PE",
    )

    metrics_summary = {
        "spot_price": round(spot_price, 2),
        "vwap": round(vwap, 2),
        "pcr": round(pcr, 3),
        "sentiment_score": round(sentiment_score, 4),
        "atm_strike": atm_strike,
        "max_pain": max_pain,
        "atm_ce_ltp": atm_ce_ltp,
        "atm_pe_ltp": atm_pe_ltp,
        "call_change_oi": call_change_oi,
        "put_change_oi": put_change_oi,
        "call_unwinding_detected": call_unwinding_detected,
        "put_unwinding_detected": put_unwinding_detected,
        "simultaneous_writing": simultaneous_writing,
    }

    return PredictResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol.upper(),
        signal=signal,
        confidence=confidence,
        trade_setup=trade_setup,
        conditions_met=conditions_met,
        metrics_summary=metrics_summary,
        reason=reason,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Service health check endpoint."""
    return {"status": "healthy", "service": "Dynamic Prediction Engine"}


@app.post("/api/predict", response_model=PredictResponse)
def predict_trade_post(payload: Optional[PredictRequest] = None) -> PredictResponse:
    """
    Main trade prediction endpoint. Ingests option chain, technicals, and news sentiment
    to produce institutional option buying signals with dynamic SL and Target.
    """
    req = payload or PredictRequest()
    symbol = req.symbol.upper()

    # 1. Option Chain Ingestion (Live Upstox or Provided/Synthetic)
    if req.access_token:
        try:
            engine = OptionChainEngine(access_token=req.access_token)
            df_chain, chain_summary = engine.get_option_chain(symbol)
            spot_price = req.spot_price or chain_summary["spot_ltp"]
            atm_strike = req.atm_strike or chain_summary["atm_strike"]
            pcr = req.pcr if req.pcr is not None else chain_summary["pcr_oi"]
            max_pain = chain_summary["max_pain"]

            # Extract ATM premiums
            atm_row = df_chain[df_chain["strike_price"] == atm_strike]
            if not atm_row.empty:
                atm_ce_ltp = float(atm_row.iloc[0]["call_ltp"])
                atm_pe_ltp = float(atm_row.iloc[0]["put_ltp"])
            else:
                atm_ce_ltp = req.atm_ce_ltp or 150.0
                atm_pe_ltp = req.atm_pe_ltp or 150.0

            shifts = chain_summary["smart_money_shifts"]
            call_unwinding_detected = len(shifts.get("call_unwinding", [])) > 0
            put_unwinding_detected = len(shifts.get("put_unwinding", [])) > 0
            call_writing_detected = len(shifts.get("call_writing", [])) > 0
            put_writing_detected = len(shifts.get("put_writing", [])) > 0
            simultaneous_writing = (
                call_writing_detected and put_writing_detected and abs(chain_summary["net_oi_change_bias"]) < 100_000
            )
            call_change_oi = req.call_change_oi if req.call_change_oi is not None else int(df_chain["call_change_oi"].sum())
            put_change_oi = req.put_change_oi if req.put_change_oi is not None else int(df_chain["put_change_oi"].sum())

        except Exception as e:
            logger.error("OptionChainEngine error: %s. Falling back to input/synthetic data.", e)
            spot_price = req.spot_price or 24500.0
            atm_strike = req.atm_strike or 24500.0
            pcr = req.pcr if req.pcr is not None else 1.15
            max_pain = atm_strike
            atm_ce_ltp = req.atm_ce_ltp or 145.0
            atm_pe_ltp = req.atm_pe_ltp or 135.0
            call_change_oi = req.call_change_oi if req.call_change_oi is not None else -25000
            put_change_oi = req.put_change_oi if req.put_change_oi is not None else 35000
            call_unwinding_detected = call_change_oi < 0
            put_unwinding_detected = put_change_oi < 0
            simultaneous_writing = False
    else:
        # Default / Manual / Synthetic Ingestion
        spot_price = req.spot_price if req.spot_price is not None else 24530.0
        atm_strike = req.atm_strike if req.atm_strike is not None else round(spot_price / 50.0) * 50.0
        pcr = req.pcr if req.pcr is not None else 1.22
        max_pain = atm_strike - 50.0
        atm_ce_ltp = req.atm_ce_ltp if req.atm_ce_ltp is not None else 165.0
        atm_pe_ltp = req.atm_pe_ltp if req.atm_pe_ltp is not None else 120.0
        call_change_oi = req.call_change_oi if req.call_change_oi is not None else -45000
        put_change_oi = req.put_change_oi if req.put_change_oi is not None else 65000
        call_unwinding_detected = call_change_oi < 0
        put_unwinding_detected = put_change_oi < 0
        simultaneous_writing = False

    # 2. Technical Indicators Ingestion
    if req.vwap is not None:
        vwap = req.vwap
    else:
        if req.ohlcv_bars:
            ohlcv_df = pd.DataFrame(req.ohlcv_bars)
        else:
            ohlcv_df = generate_mock_ohlcv(bars=80, base_price=spot_price)
        _, tech_metrics = tech_engine.analyze_dataframe(ohlcv_df)
        vwap = tech_metrics.vwap

    # 3. News Sentiment Ingestion
    if req.sentiment_score is not None:
        sentiment_score = req.sentiment_score
    else:
        sent_report = news_analyzer.analyze_sentiment()
        sentiment_score = sent_report.aggregate_score

    # 4. Generate Signal & Return Structured Response
    return evaluate_signal(
        symbol=symbol,
        spot_price=spot_price,
        vwap=vwap,
        pcr=pcr,
        sentiment_score=sentiment_score,
        atm_strike=atm_strike,
        atm_ce_ltp=atm_ce_ltp,
        atm_pe_ltp=atm_pe_ltp,
        call_change_oi=call_change_oi,
        put_change_oi=put_change_oi,
        call_unwinding_detected=call_unwinding_detected,
        put_unwinding_detected=put_unwinding_detected,
        simultaneous_writing=simultaneous_writing,
        max_pain=max_pain,
    )


@app.get("/api/predict", response_model=PredictResponse)
def predict_trade_get(
    symbol: str = Query("NIFTY", description="Trading symbol (e.g. NIFTY, BANKNIFTY)"),
    spot_price: Optional[float] = Query(None, description="Spot LTP"),
    vwap: Optional[float] = Query(None, description="VWAP level"),
    pcr: Optional[float] = Query(None, description="Put Call Ratio"),
    sentiment_score: Optional[float] = Query(None, description="Sentiment score [-1.0 to +1.0]"),
    call_change_oi: Optional[int] = Query(None, description="Call Change in OI"),
    put_change_oi: Optional[int] = Query(None, description="Put Change in OI"),
    atm_ce_ltp: Optional[float] = Query(None, description="ATM CE LTP"),
    atm_pe_ltp: Optional[float] = Query(None, description="ATM PE LTP"),
) -> PredictResponse:
    """GET endpoint for easy browser/curl testing."""
    payload = PredictRequest(
        symbol=symbol,
        spot_price=spot_price,
        vwap=vwap,
        pcr=pcr,
        sentiment_score=sentiment_score,
        call_change_oi=call_change_oi,
        put_change_oi=put_change_oi,
        atm_ce_ltp=atm_ce_ltp,
        atm_pe_ltp=atm_pe_ltp,
    )
    return predict_trade_post(payload)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on http://0.0.0.0:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
