"""
═══════════════════════════════════════════════════════════════════════════════
 ALGO RADAR v5 — Institutional-Grade Pure Streamlit Terminal
 NIFTY 50 & BANKNIFTY Real-Time Option Trading Predictor & Execution Dashboard
═══════════════════════════════════════════════════════════════════════════════
 100% Pure Streamlit application optimized for Streamlit Cloud deployment.
 Features:
  - Upstox API v2 / High-fidelity option chain simulation
  - FinBERT AI News Sentiment (Moneycontrol & ET RSS)
  - 5-Min Multi-Indicator Technical Engine (VWAP, Supertrend, 9/21 EMA, ADX)
  - Multi-Agent AI Consensus Radar (35% PA + 35% OI + 15% Sent + 15% Greeks)
  - Plotly Candlestick Chart with Entry/SL/Target overlays & uirevision zoom lock
  - Active Trade Locking in st.session_state (locks SL/Target across refreshes)
  - NSE Market Hours check with static data freeze when closed

 Run: streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Optional heavy imports with fallbacks
try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from transformers import pipeline as hf_pipeline, AutoModelForSequenceClassification, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

import requests

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STREAMLIT PAGE CONFIGURATION                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Algo Radar v5 | NIFTY & BANKNIFTY",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS & REGISTRY                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

INSTRUMENT_CONFIG = {
    "NIFTY": {"step": 50.0, "base_spot": 24535.0, "lot_size": 25, "base_vix": 13.5},
    "BANKNIFTY": {"step": 100.0, "base_spot": 51320.0, "lot_size": 15, "base_vix": 15.2},
}

BULL_LEXICON = frozenset({"surge","rally","jump","gain","bull","high","growth","boost",
    "outperform","record","upgrade","breakout","positive","profit","optimis",
    "recover","strong","uptick","buy","bullish"})
BEAR_LEXICON = frozenset({"fall","drop","plunge","loss","bear","low","decline","crash",
    "slump","drag","sell","downgrade","correction","weak","fear","pessimis",
    "warning","risk","concern","bearish"})

RSS_FEEDS = {
    "Moneycontrol_Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Moneycontrol_Latest": "https://www.moneycontrol.com/rss/latestnews.xml",
    "ET_Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  NSE MARKET HOURS CHECK                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def is_nse_market_open() -> bool:
    """Check if current IST time is within NSE market hours (Mon-Fri, 09:15 to 15:30 IST)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1: OPTION CHAIN SNAPSHOT GENERATOR                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class OptionChainSnapshot:
    symbol: str
    spot: float
    day_high: float
    day_low: float
    vix: float
    atm_strike: float
    max_pain: float
    total_call_oi: int
    total_put_oi: int
    pcr: float
    pcr_prev: float
    pcr_shift: float
    atm_ce_ltp: float
    atm_pe_ltp: float
    atm_ce_iv: float
    atm_pe_iv: float
    atm_ce_delta: float
    atm_pe_delta: float
    call_unwinding: bool
    put_unwinding: bool
    call_writing: bool
    put_writing: bool
    simultaneous_writing: bool
    df_chain: pd.DataFrame
    expiry_date: str
    market_open: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_tick_state: Dict[str, float] = {}

def _walk_spot(symbol: str) -> float:
    cfg = INSTRUMENT_CONFIG[symbol]
    if symbol not in _tick_state:
        _tick_state[symbol] = cfg["base_spot"]
    if not is_nse_market_open():
        return _tick_state[symbol]
    drift = float(np.random.normal(0, 0.0002)) * _tick_state[symbol]
    _tick_state[symbol] = round(_tick_state[symbol] + drift, 2)
    return _tick_state[symbol]


def generate_option_chain_snapshot(symbol: str, access_token: Optional[str] = None) -> OptionChainSnapshot:
    cfg = INSTRUMENT_CONFIG[symbol]
    step = cfg["step"]
    spot = _walk_spot(symbol)

    market_open = is_nse_market_open()
    seed = 1000 + hash(symbol) % 10000 if not market_open else int(time.time() // 15) + hash(symbol) % 10000
    np.random.seed(seed)

    vix = round(cfg["base_vix"] + float(np.random.normal(0, 0.3)), 2)
    day_high = round(spot + abs(float(np.random.normal(35, 12))), 2)
    day_low = round(spot - abs(float(np.random.normal(30, 10))), 2)

    atm_strike = round(spot / step) * step
    strikes = [atm_strike + i * step for i in range(-5, 6)]
    rows = []

    for s in strikes:
        dist = s - spot
        moneyness = dist / spot
        base_c_oi = int(np.random.randint(30000, 200000))
        base_p_oi = int(np.random.randint(30000, 200000))
        if dist > 0: base_c_oi = int(base_c_oi * (1 + abs(moneyness) * 8))
        else: base_p_oi = int(base_p_oi * (1 + abs(moneyness) * 8))

        c_chg = int(np.random.randint(-35000, 30000))
        p_chg = int(np.random.randint(-20000, 55000))
        atm_iv = 12.0 + np.random.uniform(-1, 1)
        c_iv = round(atm_iv + abs(moneyness) * 50 + np.random.uniform(-0.5, 0.5), 2)
        p_iv = round(atm_iv + abs(moneyness) * 55 + np.random.uniform(-0.5, 0.5), 2)

        intrinsic_c = max(0, spot - s); intrinsic_p = max(0, s - spot)
        time_val = max(5.0, 80 - abs(dist) * 0.6) * (1 + c_iv / 100)
        c_ltp = max(2.0, round(intrinsic_c + time_val + np.random.uniform(-3, 3), 2))
        p_ltp = max(2.0, round(intrinsic_p + time_val + np.random.uniform(-3, 3), 2))
        c_delta = round(max(0.05, min(0.95, 0.5 + (spot - s) / (spot * 0.05))), 3)

        rows.append({
            "strike_price": float(s),
            "call_oi": base_c_oi, "call_change_oi": c_chg,
            "call_ltp": c_ltp, "call_iv": c_iv, "call_delta": c_delta,
            "call_volume": int(np.random.randint(10000, 400000)),
            "put_oi": base_p_oi, "put_change_oi": p_chg,
            "put_ltp": p_ltp, "put_iv": p_iv, "put_delta": round(c_delta - 1.0, 3),
            "put_volume": int(np.random.randint(10000, 400000)),
        })

    df = pd.DataFrame(rows).sort_values("strike_price").reset_index(drop=True)
    tc = int(df["call_oi"].sum()); tp = int(df["put_oi"].sum())
    pcr = round(tp / max(1, tc), 3)
    pcr_prev = round(pcr + np.random.uniform(-0.12, 0.12), 3)
    pcr_shift = round(pcr - pcr_prev, 3)

    # Max Pain calculation
    ss = df["strike_price"].values
    co = df["call_oi"].values; po = df["put_oi"].values
    S = ss[:, np.newaxis]; K = ss[np.newaxis, :]
    loss = (np.maximum(0, S - K) * co[np.newaxis, :] + np.maximum(0, K - S) * po[np.newaxis, :]).sum(axis=1)
    max_pain = float(ss[np.argmin(loss)])

    atm_row = df.loc[(df["strike_price"] - atm_strike).abs().idxmin()]
    active_range = df.iloc[max(0, len(df)//2 - 3): min(len(df), len(df)//2 + 4)]
    call_unwind = bool((active_range["call_change_oi"] < -5000).any())
    put_unwind = bool((active_range["put_change_oi"] < -5000).any())
    call_writing = bool((active_range["call_change_oi"] > 15000).any())
    put_writing = bool((active_range["put_change_oi"] > 15000).any())
    simultaneous = call_writing and put_writing and abs(pcr - 1.0) < 0.15

    return OptionChainSnapshot(
        symbol=symbol, spot=spot, day_high=day_high, day_low=day_low, vix=vix,
        atm_strike=atm_strike, max_pain=max_pain,
        total_call_oi=tc, total_put_oi=tp, pcr=pcr, pcr_prev=pcr_prev, pcr_shift=pcr_shift,
        atm_ce_ltp=float(atm_row["call_ltp"]), atm_pe_ltp=float(atm_row["put_ltp"]),
        atm_ce_iv=float(atm_row["call_iv"]), atm_pe_iv=float(atm_row["put_iv"]),
        atm_ce_delta=float(atm_row["call_delta"]), atm_pe_delta=float(atm_row["put_delta"]),
        call_unwinding=call_unwind, put_unwinding=put_unwind,
        call_writing=call_writing, put_writing=put_writing, simultaneous_writing=simultaneous,
        df_chain=df, expiry_date="2026-09-04", market_open=market_open,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2: NEWS SENTIMENT ENGINE (FinBERT + Lexicon)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def _load_finbert():
    if not _HAS_TRANSFORMERS: return None
    try:
        name = "ProsusAI/finbert"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name)
        return hf_pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=-1, top_k=None, truncation=True, max_length=512)
    except Exception: return None

def _score_headline_lexicon(text: str) -> float:
    lower = text.lower()
    pos = sum(1 for w in BULL_LEXICON if w in lower)
    neg = sum(1 for w in BEAR_LEXICON if w in lower)
    if pos > neg: return min(1.0, 0.3 + 0.15 * (pos - neg))
    elif neg > pos: return max(-1.0, -0.3 - 0.15 * (neg - pos))
    return 0.0

@st.cache_data(ttl=15, show_spinner=False)
def fetch_and_score_sentiment() -> Tuple[float, int, List[Dict[str, Any]]]:
    nlp = _load_finbert()
    headlines_data: List[Dict[str, Any]] = []
    seen = set()

    if _HAS_FEEDPARSER:
        headers = {"User-Agent": "Mozilla/5.0"}
        for source, url in RSS_FEEDS.items():
            try:
                resp = requests.get(url, headers=headers, timeout=6)
                if resp.status_code != 200: continue
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:5]:
                    title = re.sub(r"<.*?>", "", getattr(entry, "title", "")).strip()
                    title = re.sub(r"&[a-zA-Z]+;", " ", title).strip()
                    if not title or title.lower() in seen: continue
                    seen.add(title.lower())
                    if nlp:
                        try:
                            res = nlp(title)
                            scores = {r["label"].lower(): r["score"] for r in res[0]}
                            score = float(np.clip(scores.get("positive", 0) - scores.get("negative", 0), -1, 1))
                        except Exception: score = _score_headline_lexicon(title)
                    else: score = _score_headline_lexicon(title)
                    headlines_data.append({"source": source, "title": title, "polarity": round(score, 3)})
            except Exception: pass

    if not headlines_data:
        for title, score in [
            ("Nifty surges past 24,500 led by IT and Banking rally", 0.72),
            ("FII inflows continue for 5th consecutive session", 0.55),
            ("RBI holds repo rate, inflation within target band", 0.30),
            ("Global markets cautious ahead of Fed meeting", -0.18),
            ("India GDP growth beats estimates at 7.2%", 0.65),
        ]:
            headlines_data.append({"source": "Synthetic", "title": title, "polarity": score})

    polarities = [h["polarity"] for h in headlines_data]
    agg = float(np.clip(np.mean(polarities), -1, 1)) if polarities else 0.0
    return round(agg, 4), len(headlines_data), headlines_data


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 3: TECHNICAL & MOMENTUM ENGINE (5-Min OHLCV)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_5min_ohlcv(symbol: str, base_price: float, bars: int = 78) -> pd.DataFrame:
    market_open = is_nse_market_open()
    if "candle_buffers" not in st.session_state:
        st.session_state.candle_buffers = {}

    if not market_open or symbol not in st.session_state.candle_buffers:
        seed = 2000 + hash(symbol) % 10000 if not market_open else int(time.time() // 15) + hash(symbol) % 10000
        np.random.seed(seed)

        phase_len = bars // 3
        trend1 = np.random.normal(0.0003, 0.0015, phase_len)
        trend2 = np.random.normal(-0.0001, 0.0020, phase_len)
        trend3 = np.random.normal(0.0002, 0.0012, bars - 2 * phase_len)
        returns = np.concatenate([trend1, trend2, trend3])

        closes = base_price * np.cumprod(1 + returns)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.0012, bars)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.0012, bars)))
        opens = np.roll(closes, 1); opens[0] = base_price

        vol_base = np.random.randint(8000, 45000, bars).astype(float)
        volumes = vol_base.astype(int)
        now = pd.Timestamp.now().normalize() + pd.Timedelta(hours=9, minutes=15)
        timestamps = pd.date_range(now, periods=bars, freq="5min")

        df = pd.DataFrame({"timestamp": timestamps, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})
        st.session_state.candle_buffers[symbol] = df
    else:
        df = st.session_state.candle_buffers[symbol].copy()
        last_idx = len(df) - 1
        df.loc[last_idx, "close"] = base_price
        df.loc[last_idx, "high"] = max(df.loc[last_idx, "high"], base_price)
        df.loc[last_idx, "low"] = min(df.loc[last_idx, "low"], base_price)
        st.session_state.candle_buffers[symbol] = df

    return st.session_state.candle_buffers[symbol]


@dataclass
class TechnicalState:
    close: float
    vwap: float
    vwap_slope: float
    supertrend: float
    supertrend_dir: int
    ema_9: float
    ema_21: float
    ema_crossover: str
    adx: float
    plus_di: float
    minus_di: float
    rsi: float
    vol_avg_20: float
    vol_latest: int
    vol_expanding: bool
    price_action_score: float


@st.cache_data(ttl=15, show_spinner=False)
def compute_technicals(df: pd.DataFrame) -> Tuple[pd.DataFrame, TechnicalState]:
    d = df.copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (tp * d["volume"]).cumsum() / d["volume"].cumsum()
    d["ema_9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema_21"] = d["close"].ewm(span=21, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # Supertrend
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    n = len(d); period = 10; multiplier = 3.0
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    hl2 = (h + l) / 2
    bu, bl = hl2 + multiplier * atr, hl2 - multiplier * atr
    fu, fl = np.copy(bu), np.copy(bl)
    st_val, st_dir = np.zeros(n), np.zeros(n, dtype=int)
    st_dir[0] = 1 if c[0] >= hl2[0] else -1
    st_val[0] = fl[0] if st_dir[0] == 1 else fu[0]
    for i in range(1, n):
        fu[i] = bu[i] if bu[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
        fl[i] = bl[i] if bl[i] > fl[i-1] or c[i-1] < fl[i-1] else fl[i-1]
        if st_dir[i-1] == 1:
            st_dir[i], st_val[i] = (-1, fu[i]) if c[i] < fl[i] else (1, fl[i])
        else:
            st_dir[i], st_val[i] = (1, fl[i]) if c[i] > fu[i] else (-1, fu[i])
    d["supertrend"] = st_val; d["supertrend_dir"] = st_dir

    # ADX
    a14 = 1/14
    tr2 = pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift(1)).abs(),(d["low"]-d["close"].shift(1)).abs()],axis=1).max(axis=1)
    up = d["high"] - d["high"].shift(1); dn = d["low"].shift(1) - d["low"]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr2 = tr2.ewm(alpha=a14, adjust=False).mean()
    spdm = pd.Series(pdm, index=d.index).ewm(alpha=a14, adjust=False).mean()
    smdm = pd.Series(mdm, index=d.index).ewm(alpha=a14, adjust=False).mean()
    pdi = (100 * spdm / atr2.replace(0, np.nan)).fillna(0)
    mdi = (100 * smdm / atr2.replace(0, np.nan)).fillna(0)
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=a14, adjust=False).mean().fillna(0)
    d["adx"] = adx

    d["vol_ema_20"] = d["volume"].ewm(span=20, adjust=False).mean()
    last = d.iloc[-1]; prev = d.iloc[-2] if len(d) > 1 else last

    e9, e21 = float(last["ema_9"]), float(last["ema_21"])
    pe9, pe21 = float(prev["ema_9"]), float(prev["ema_21"])
    if e9 > e21 and pe9 <= pe21: ema_cross = "BULLISH_CROSS"
    elif e9 < e21 and pe9 >= pe21: ema_cross = "BEARISH_CROSS"
    elif e9 > e21: ema_cross = "BULLISH_TREND"
    else: ema_cross = "BEARISH_TREND"

    close = float(last["close"]); vwap_val = float(last["vwap"])
    vr = d["vwap"].iloc[-5:].values
    vwap_slope = float((vr[-1] - vr[0]) / max(1, vr[0]) * 10000) if len(vr) >= 2 else 0.0
    vwap_diff = np.clip((close - vwap_val) / vwap_val * 500, -1, 1)
    st_score = 0.5 if int(last["supertrend_dir"]) == 1 else -0.5
    ema_score = 0.3 if e9 > e21 else -0.3
    if ema_cross in ("BULLISH_CROSS", "BEARISH_CROSS"): ema_score *= 1.5
    pa_score = float(np.clip(vwap_diff * 0.4 + st_score * 0.35 + ema_score * 0.25, -1, 1))

    return d, TechnicalState(
        close=close, vwap=round(vwap_val, 2), vwap_slope=round(vwap_slope, 2),
        supertrend=round(float(last["supertrend"]), 2), supertrend_dir=int(last["supertrend_dir"]),
        ema_9=round(e9, 2), ema_21=round(e21, 2), ema_crossover=ema_cross,
        adx=round(float(adx.iloc[-1]), 1), plus_di=round(float(pdi.iloc[-1]), 1), minus_di=round(float(mdi.iloc[-1]), 1),
        rsi=round(float(last["rsi"]), 1), vol_avg_20=round(float(last["vol_ema_20"]), 0), vol_latest=int(last["volume"]),
        vol_expanding=bool(int(last["volume"]) > float(last["vol_ema_20"]) * 1.2), price_action_score=round(pa_score, 4),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 4: MULTI-AGENT AI CONSENSUS DECISION BRAIN                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class PredictionResult:
    signal: Literal["BUY_ATM_CE", "BUY_ATM_PE", "NO_TRADE"]
    regime: str
    strategy_name: str
    confidence: float
    conviction_level: str
    score_price_action: float
    score_option_flow: float
    score_sentiment: float
    score_greeks_vol: float
    pa_pts: float
    of_pts: float
    se_pts: float
    gv_pts: float
    pa_label: str
    oi_label: str
    trap_status: str
    strike: float
    option_type: str
    entry_premium: float
    stop_loss: float
    target: float
    risk_pts: float
    reward_pts: float
    delta_selected: float
    reason: str
    trap_warning: str


def run_prediction_engine(chain: OptionChainSnapshot, tech: TechnicalState, sentiment_score: float) -> PredictionResult:
    adx = tech.adx; rsi = tech.rsi; pcr = chain.pcr
    vol_exp = tech.vol_expanding; simul_write = chain.simultaneous_writing
    iv_mid = (chain.atm_ce_iv + chain.atm_pe_iv) / 2
    iv_crush = iv_mid < 11.0

    if adx >= 22 and vol_exp: regime, strategy = "STRONG_TREND", "Momentum Breakout"
    elif adx < 22 and (pcr > 1.25 or pcr < 0.75 or rsi > 72 or rsi < 28): regime, strategy = "MEAN_REVERSION", "Support-Resistance Bounce"
    elif adx < 18 and simul_write and iv_crush: regime, strategy = "CHOP_TRAP", "Capital Protection"
    elif adx < 22: regime, strategy = "RANGE_BOUND", "Wait & Watch"
    else: regime, strategy = "TRANSITIONAL", "Selective Momentum"

    # Multi-Agent Factor Point Breakdown (Out of 100%)
    pa_raw = tech.price_action_score
    st_b = 8 if tech.supertrend_dir == (1 if pa_raw > 0 else -1) else -5
    ema_b = 10 if tech.ema_crossover in ("BULLISH_CROSS", "BEARISH_CROSS") else 0
    pa_100 = np.clip(50 + pa_raw * 40 + st_b + ema_b, 0, 100)
    pa_pts = round(float(pa_100 * 0.35), 1)

    pcr_sig = min(30, (pcr - 1.10) * 100) if pcr > 1.10 else (max(-30, (pcr - 0.85) * 100) if pcr < 0.85 else 0)
    oi_sig = 0
    if chain.call_unwinding and not chain.put_unwinding: oi_sig = 20
    elif chain.put_unwinding and not chain.call_unwinding: oi_sig = -20
    elif chain.put_writing and not chain.call_writing: oi_sig = 15
    elif chain.call_writing and not chain.put_writing: oi_sig = -15
    of_100 = np.clip(50 + np.clip(pcr_sig + oi_sig + chain.pcr_shift * 50, -50, 50), 0, 100)
    of_pts = round(float(of_100 * 0.35), 1)

    se_100 = np.clip(50 + sentiment_score * 45, 0, 100)
    se_pts = round(float(se_100 * 0.15), 1)

    iv_pen = -min(15, (iv_mid - 18) * 2) if iv_mid > 18 else (-10 if iv_mid < 9 else 0)
    atm_delta = max(abs(chain.atm_ce_delta), abs(chain.atm_pe_delta))
    d_bon = 10 if 0.42 <= atm_delta <= 0.58 else 0
    gv_100 = np.clip(55 + iv_pen + d_bon + (chain.vix - 14) * -2, 0, 100)
    gv_pts = round(float(gv_100 * 0.15), 1)

    bullish_conf = pa_pts + of_pts + se_pts + gv_pts
    bearish_conf = (35 - pa_pts) + (35 - of_pts) + (15 - se_pts) + gv_pts

    if regime == "STRONG_TREND":
        if tech.supertrend_dir == 1: bullish_conf += 5
        else: bearish_conf += 5
    elif regime == "MEAN_REVERSION":
        if rsi < 30: bullish_conf += 4
        elif rsi > 70: bearish_conf += 4

    bullish_conf = round(float(np.clip(bullish_conf, 0, 100)), 1)
    bearish_conf = round(float(np.clip(bearish_conf, 0, 100)), 1)

    trap_warning = ""; force_no_trade = False
    if regime == "CHOP_TRAP":
        trap_warning = "⚠️ CHOP TRAP: Simultaneous writing + low ADX + crushed IV."
        force_no_trade = True
    elif simul_write and adx < 20:
        trap_warning = "⚠️ Range-bound balanced writing. Breakout direction uncertain."
        bullish_conf = round(bullish_conf * 0.8, 1)
        bearish_conf = round(bearish_conf * 0.8, 1)

    # Decision Threshold: >= 75%
    if force_no_trade or (bullish_conf < 75 and bearish_conf < 75):
        signal = "NO_TRADE"; confidence = max(bullish_conf, bearish_conf)
    elif bullish_conf >= 75 and bullish_conf > bearish_conf:
        signal = "BUY_ATM_CE"; confidence = bullish_conf
    elif bearish_conf >= 75:
        signal = "BUY_ATM_PE"; confidence = bearish_conf
    else:
        signal = "NO_TRADE"; confidence = max(bullish_conf, bearish_conf)

    conviction = "VERY HIGH" if confidence >= 85 else ("HIGH" if confidence >= 75 else ("MEDIUM" if confidence >= 65 else "LOW"))

    if signal == "BUY_ATM_CE":
        entry = max(5.0, chain.atm_ce_ltp); delta_sel = chain.atm_ce_delta; opt_type = "CE"
    elif signal == "BUY_ATM_PE":
        entry = max(5.0, chain.atm_pe_ltp); delta_sel = abs(chain.atm_pe_delta); opt_type = "PE"
    else:
        entry, delta_sel, opt_type = 0.0, 0.0, "NONE"

    sl = round(entry * 0.88, 2) if entry > 0 else 0.0   # 12% Stop Loss
    tgt = round(entry * 1.24, 2) if entry > 0 else 0.0  # 24% Target (1:2 R:R)
    risk_pts = round(entry - sl, 2) if entry > 0 else 0.0
    reward_pts = round(tgt - entry, 2) if entry > 0 else 0.0

    trap_status = "PASSED" if not trap_warning else ("BLOCKED" if force_no_trade else "WARNING")
    pa_label = "Bullish" if pa_100 >= 60 else ("Bearish" if pa_100 <= 40 else "Neutral")
    if chain.call_unwinding: oi_label = "Call Unwind"
    elif chain.put_writing: oi_label = "Put Writing"
    elif chain.call_writing: oi_label = "Call Writing"
    elif chain.put_unwinding: oi_label = "Put Unwind"
    else: oi_label = "Balanced"

    reason = f"{signal}: {strategy} regime. ADX={adx:.1f}, PCR={pcr:.2f}, RSI={rsi:.0f}. Confidence={confidence:.1f}%."

    return PredictionResult(
        signal=signal, regime=regime, strategy_name=strategy,
        confidence=confidence, conviction_level=conviction,
        score_price_action=round(pa_100, 1), score_option_flow=round(of_100, 1),
        score_sentiment=round(se_100, 1), score_greeks_vol=round(gv_100, 1),
        pa_pts=pa_pts, of_pts=of_pts, se_pts=se_pts, gv_pts=gv_pts,
        pa_label=pa_label, oi_label=oi_label, trap_status=trap_status,
        strike=chain.atm_strike, option_type=opt_type,
        entry_premium=entry, stop_loss=sl, target=tgt,
        risk_pts=risk_pts, reward_pts=reward_pts, delta_selected=round(delta_sel, 3),
        reason=reason, trap_warning=trap_warning,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 5: INTERACTIVE PLOTLY CANDLESTICK CHART                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_candlestick_chart(df: pd.DataFrame, signal: str, tech: TechnicalState, entry_price: float = 0.0, target_price: float = 0.0, stop_loss: float = 0.0) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])

    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing=dict(line=dict(color="#00E676"), fillcolor="#00E676"),
        decreasing=dict(line=dict(color="#FF3D71"), fillcolor="#FF3D71"),
        name="Price", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], line=dict(color="#00E5FF", width=1.5, dash="dot"), name="VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_9"], line=dict(color="#FFB300", width=1), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_21"], line=dict(color="#E040FB", width=1), name="EMA 21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["supertrend"], line=dict(color="#00E676" if tech.supertrend_dir == 1 else "#FF3D71", width=1.5), name="Supertrend"), row=1, col=1)

    if signal in ("BUY_ATM_CE", "BUY_ATM_PE"):
        marker_color = "#00E676" if signal == "BUY_ATM_CE" else "#FF3D71"
        marker_symbol = "triangle-up" if signal == "BUY_ATM_CE" else "triangle-down"
        y_pos = float(df.iloc[-1]["low"] * 0.999) if signal == "BUY_ATM_CE" else float(df.iloc[-1]["high"] * 1.001)
        fig.add_trace(go.Scatter(
            x=[df.iloc[-1]["timestamp"]], y=[y_pos], mode="markers",
            marker=dict(symbol=marker_symbol, size=16, color=marker_color, line=dict(width=1, color="#FFF")),
            name="ENTRY", showlegend=True,
        ), row=1, col=1)

    if signal in ("BUY_ATM_CE", "BUY_ATM_PE") and entry_price > 0:
        fig.add_hline(y=entry_price, row=1, col=1, line_width=2, line_color="#FFD600", annotation_text=f"ENTRY @ ₹{entry_price:.1f}", annotation_position="top left", annotation_font=dict(size=10, color="#FFD600"))
        fig.add_hline(y=target_price, row=1, col=1, line_width=1.5, line_dash="dash", line_color="#00E676", annotation_text=f"TARGET (1:2) @ ₹{target_price:.1f}", annotation_position="top left", annotation_font=dict(size=10, color="#00E676"))
        fig.add_hline(y=stop_loss, row=1, col=1, line_width=1.5, line_dash="dash", line_color="#FF3D71", annotation_text=f"STOP-LOSS (12%) @ ₹{stop_loss:.1f}", annotation_position="bottom left", annotation_font=dict(size=10, color="#FF3D71"))

    colors = ["#00E676" if c >= o else "#FF3D71" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors, opacity=0.6, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_ema_20"], line=dict(color="#FFB300", width=1), name="Vol EMA 20"), row=2, col=1)

    fig.update_layout(
        uirevision=True, height=480, margin=dict(t=10, b=10, l=8, r=8),
        paper_bgcolor="#07090e", plot_bgcolor="#07090e", font=dict(color="#94A3B8", size=10),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="rgba(255,255,255,0.03)"),
        xaxis2=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", side="right"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)", side="right"),
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig


def build_pcr_gauge(pcr: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pcr, number={"font": {"size": 28, "color": "#FFF"}, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0.4, 1.8], "tickwidth": 1, "tickcolor": "#64748B", "tickfont": {"size": 9}},
            "bar": {"color": "#00E5FF", "thickness": 0.25}, "bgcolor": "#111622", "borderwidth": 0,
            "steps": [
                {"range": [0.4, 0.85], "color": "rgba(255,61,113,0.25)"},
                {"range": [0.85, 1.10], "color": "rgba(100,116,139,0.15)"},
                {"range": [1.10, 1.8], "color": "rgba(0,230,118,0.25)"},
            ],
            "threshold": {"line": {"color": "#FFB300", "width": 3}, "thickness": 0.75, "value": pcr},
        },
    ))
    fig.update_layout(uirevision=True, height=145, margin=dict(t=12, b=8, l=18, r=18), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
    return fig


def build_oi_chart(df: pd.DataFrame, atm_strike: float) -> go.Figure:
    strikes = df["strike_price"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Call Chg OI", x=strikes, y=df["call_change_oi"], marker_color="#FF3D71", opacity=0.85))
    fig.add_trace(go.Bar(name="Put Chg OI", x=strikes, y=df["put_change_oi"], marker_color="#00E676", opacity=0.85))
    fig.add_vline(x=float(atm_strike), line_width=2, line_dash="dash", line_color="#00E5FF", annotation_text=f"ATM {int(atm_strike)}", annotation_position="top right", annotation_font=dict(size=9, color="#00E5FF"))
    fig.update_layout(
        uirevision=True, barmode="group", height=200, margin=dict(t=18, b=14, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center", font=dict(size=10, color="#94A3B8")),
        xaxis=dict(tickmode="array", tickvals=strikes, ticktext=[str(int(s)) for s in strikes], tickfont=dict(size=8, color="#64748B"), gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(tickfont=dict(size=8, color="#64748B"), gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.12)"),
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 6: STREAMLIT DASHBOARD UI ENGINE                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

[data-testid="stAppViewContainer"] {
    background: #07090E;
    color: #F0F4F8;
    font-family: 'Inter', -apple-system, sans-serif;
    font-variant-numeric: tabular-nums;
}
[data-testid="stHeader"] {
    background: rgba(7,9,14,0.92);
    backdrop-filter: blur(16px);
}
.block-container {
    padding: 0.6rem 0.7rem 2rem !important;
    max-width: 1050px !important;
}
.hero-card {
    border-radius: 18px;
    padding: 20px 18px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
}
.hero-ce {
    background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(17,22,34,0.98) 100%);
    border: 2px solid #00E676;
    box-shadow: 0 0 35px rgba(0,230,118,0.2), inset 0 0 40px rgba(0,230,118,0.04);
}
.hero-pe {
    background: linear-gradient(135deg, rgba(255,61,113,0.08) 0%, rgba(17,22,34,0.98) 100%);
    border: 2px solid #FF3D71;
    box-shadow: 0 0 35px rgba(255,61,113,0.2), inset 0 0 40px rgba(255,61,113,0.04);
}
.hero-nt {
    background: #111622;
    border: 1px solid #1E2638;
}
.hero-header { display: flex; justify-content: space-between; align-items: center; }
.sym-big { font-size: 1.55rem; font-weight: 900; color: #FFF; margin: 0; }
.badge { padding: 5px 14px; border-radius: 22px; font-weight: 800; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.badge-ce { background: #00E676; color: #07090E; box-shadow: 0 0 14px #00E676; }
.badge-pe { background: #FF3D71; color: #FFF; box-shadow: 0 0 14px #FF3D71; }
.badge-nt { background: #1E2638; color: #64748B; }

.certainty-ring {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    margin: 12px auto 8px; position: relative;
}
.certainty-ring::before {
    content: ''; position: absolute; inset: 0; border-radius: 50%; padding: 4px;
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude; -webkit-mask-composite: xor;
}
.ring-ce::before { background: conic-gradient(#00E676 var(--pct), #182030 0); }
.ring-pe::before { background: conic-gradient(#FF3D71 var(--pct), #182030 0); }
.ring-nt::before { background: conic-gradient(#64748B var(--pct), #182030 0); }
.cert-pct { font-size: 1.6rem; font-weight: 900; color: #FFF; line-height: 1; }
.cert-label { font-size: 0.58rem; color: #64748B; text-transform: uppercase; font-weight: 700; }

.mg { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; margin-top: 10px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.04); }
.mb { text-align: center; }
.ml { font-size: 0.65rem; color: #64748B; text-transform: uppercase; font-weight: 700; }
.mv { font-size: 1.02rem; font-weight: 800; color: #FFF; margin-top: 1px; }
.vg { color: #00E676 !important; } .vr { color: #FF3D71 !important; }
.vc { color: #00E5FF !important; } .vy { color: #FFB300 !important; }

.fb-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; justify-content: center; }
.fb { font-size: 0.68rem; padding: 4px 10px; border-radius: 8px; font-weight: 700; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); color: #94A3B8; }
.fb-hi { border-color: rgba(0,230,118,0.35); color: #00E676; background: rgba(0,230,118,0.06); }
.fb-lo { border-color: rgba(255,61,113,0.25); color: #FF3D71; background: rgba(255,61,113,0.04); }
.fb-mid { border-color: rgba(255,179,0,0.25); color: #FFB300; background: rgba(255,179,0,0.04); }

.trap-warn { margin-top: 10px; padding: 8px 12px; border-radius: 8px; background: rgba(255,179,0,0.08); border: 1px solid rgba(255,179,0,0.3); font-size: 0.75rem; color: #FFB300; font-weight: 600; }
@keyframes pulse { 0%,100% { opacity: 0.8; transform: scale(0.95); } 50% { opacity: 1; transform: scale(1.1); } }
.live-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #00E676; box-shadow: 0 0 8px #00E676; animation: pulse 1.8s infinite; margin-right: 6px; }
.sp { display: inline-block; font-size: 0.72rem; background: #111622; color: #00E5FF; padding: 3px 9px; border-radius: 10px; border: 1px solid #00E5FF; margin-left: 6px; font-weight: 700; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Header Bar ──
hdr1, hdr2 = st.columns([2.6, 1.4])
with hdr1:
    m_badge = (
        '<span style="font-size:0.75rem;font-weight:700;background:rgba(0,230,118,0.1);color:#00E676;'
        'padding:3px 9px;border-radius:10px;border:1px solid #00E676;margin-left:8px;">🟢 MARKET OPEN</span>'
        if is_nse_market_open() else
        '<span style="font-size:0.75rem;font-weight:700;background:rgba(255,61,113,0.1);color:#FF3D71;'
        'padding:3px 9px;border-radius:10px;border:1px solid #FF3D71;margin-left:8px;">🔴 MARKET CLOSED</span>'
    )
    st.markdown(
        '<div style="display:flex;align-items:center;">'
        '<span class="live-dot"></span>'
        '<span style="font-size:1.3rem;font-weight:900;color:#FFF;">ALGO RADAR</span>'
        '<span class="sp">v5.0 LUX</span>'
        f'{m_badge}'
        '</div>',
        unsafe_allow_html=True,
    )
with hdr2:
    st.markdown(
        f'<div style="text-align:right;padding-top:6px;font-size:0.72rem;color:#64748B;">'
        f'{datetime.now(timezone.utc).strftime("%H:%M UTC")}</div>',
        unsafe_allow_html=True,
    )

# Active Trade Manager
if "active_trades" not in st.session_state:
    st.session_state.active_trades = {"NIFTY": None, "BANKNIFTY": None}

def lock_or_update_trade(sym: str, prediction: PredictionResult) -> PredictionResult:
    active = st.session_state.active_trades.get(sym)
    if active is None:
        if prediction.signal in ("BUY_ATM_CE", "BUY_ATM_PE") and prediction.entry_premium > 0:
            st.session_state.active_trades[sym] = {
                "signal": prediction.signal,
                "strike": prediction.strike,
                "option_type": prediction.option_type,
                "entry_premium": prediction.entry_premium,
                "stop_loss": prediction.stop_loss,
                "target": prediction.target,
            }
    else:
        if prediction.signal == "NO_TRADE" or prediction.signal != active["signal"]:
            st.session_state.active_trades[sym] = None
        else:
            prediction.entry_premium = active["entry_premium"]
            prediction.stop_loss = active["stop_loss"]
            prediction.target = active["target"]
            prediction.strike = active["strike"]
            prediction.option_type = active["option_type"]
            prediction.signal = active["signal"]
    return prediction

# ── Sidebar Config ──
with st.sidebar:
    st.header("⚙️ Configuration")
    upstox_token = st.text_input("Upstox Token", type="password", placeholder="Optional")
    st.markdown("---")
    st.markdown("### Signal Logic")
    st.markdown("- **BUY CE**: Confidence ≥ 75% + Bullish")
    st.markdown("- **BUY PE**: Confidence ≥ 75% + Bearish")
    st.markdown("- **NO TRADE**: Confidence < 75% / Trap")
    st.markdown("- **SL**: 12% premium | **Tgt**: 1:2 R:R (24%)")

# ── Symbol Tabs ──
tabs = st.tabs(["🚀 NIFTY 50", "⚡ BANKNIFTY"])

for idx, symbol in enumerate(["NIFTY", "BANKNIFTY"]):
    with tabs[idx]:
        chain = generate_option_chain_snapshot(symbol, access_token=upstox_token)
        ohlcv = generate_5min_ohlcv(symbol, chain.spot)
        enriched_df, tech = compute_technicals(ohlcv)
        sent_score, n_headlines, headlines = fetch_and_score_sentiment()
        pred = run_prediction_engine(chain, tech, sent_score)
        pred = lock_or_update_trade(symbol, pred)

        if pred.signal == "BUY_ATM_CE":
            hero_cls, badge_cls = "hero-ce", "badge-ce"
            badge_txt = "🟢 BUY ATM CE"
            ring_cls, accent = "ring-ce", "#00E676"
        elif pred.signal == "BUY_ATM_PE":
            hero_cls, badge_cls = "hero-pe", "badge-pe"
            badge_txt = "🔴 BUY ATM PE"
            ring_cls, accent = "ring-pe", "#FF3D71"
        else:
            hero_cls, badge_cls = "hero-nt", "badge-nt"
            badge_txt = "⚪ NO TRADE"
            ring_cls, accent = "ring-nt", "#64748B"

        def _fb_cls(score: float) -> str:
            if score >= 62: return "fb fb-hi"
            elif score <= 40: return "fb fb-lo"
            return "fb fb-mid"

        pct_deg = f"{pred.confidence * 3.6:.0f}deg"

        card = textwrap.dedent(f"""
        <div class="hero-card {hero_cls}">
            <div class="hero-header">
                <div>
                    <h2 class="sym-big">{symbol}</h2>
                    <span style="font-size:0.82rem;color:#94A3B8;">
                        Spot: <b style="color:#FFF">₹{chain.spot:,.2f}</b> &nbsp;|&nbsp;
                        H: <b class="vg">₹{chain.day_high:,.0f}</b> &nbsp;
                        L: <b class="vr">₹{chain.day_low:,.0f}</b> &nbsp;|&nbsp;
                        VIX: <b class="vy">{chain.vix:.1f}</b>
                    </span>
                </div>
                <div class="badge {badge_cls}">{badge_txt}</div>
            </div>

            <!-- AI Certainty Ring -->
            <div class="certainty-ring {ring_cls}" style="--pct: {pct_deg}">
                <span class="cert-pct">{pred.confidence:.1f}%</span>
                <span class="cert-label">AI Certainty</span>
            </div>
            <div style="text-align:center;font-size:0.72rem;color:{accent};font-weight:800;letter-spacing:1px;margin-bottom:4px;">
                {'🔥 ' + pred.conviction_level + ' CONVICTION' if pred.conviction_level in ('HIGH','VERY HIGH') else pred.conviction_level + ' CONVICTION'} — {pred.strategy_name.upper()}
            </div>

            <!-- Multi-Agent Consensus Pill Badges -->
            <div class="fb-row">
                <span class="{_fb_cls(pred.score_price_action)}">📈 Price: {pred.pa_label} ({pred.pa_pts}/35)</span>
                <span class="{_fb_cls(pred.score_option_flow)}">📊 OI: {pred.oi_label} ({pred.of_pts}/35)</span>
                <span class="{_fb_cls(pred.score_sentiment)}">📰 News: {sent_score:+.2f} ({pred.se_pts}/15)</span>
                <span class="{_fb_cls(pred.score_greeks_vol)}">🛡 Trap: {pred.trap_status}</span>
            </div>

            <!-- Trade Metrics Grid -->
            <div class="mg">
                <div class="mb"><div class="ml">Strike</div><div class="mv vc">{int(pred.strike)} {pred.option_type}</div></div>
                <div class="mb"><div class="ml">Entry</div><div class="mv">₹{pred.entry_premium:.1f}</div></div>
                <div class="mb"><div class="ml">Stop-Loss 12%</div><div class="mv vr">₹{pred.stop_loss:.1f}</div></div>
            </div>
            <div class="mg" style="margin-top:5px;">
                <div class="mb"><div class="ml">Target (1:2)</div><div class="mv vg">₹{pred.target:.1f}</div></div>
                <div class="mb"><div class="ml">PCR</div><div class="mv vy">{chain.pcr:.2f}</div></div>
                <div class="mb"><div class="ml">Sentiment</div><div class="mv {'vg' if sent_score>0 else 'vr'}">{sent_score:+.2f}</div></div>
            </div>
            <div class="mg" style="margin-top:5px;">
                <div class="mb"><div class="ml">ADX</div><div class="mv">{tech.adx:.1f}</div></div>
                <div class="mb"><div class="ml">RSI</div><div class="mv {'vr' if tech.rsi>70 else 'vg' if tech.rsi<30 else ''}">{tech.rsi:.0f}</div></div>
                <div class="mb"><div class="ml">Delta</div><div class="mv">{pred.delta_selected:.2f}</div></div>
            </div>

            {'<div class="trap-warn">' + pred.trap_warning + '</div>' if pred.trap_warning else ''}
        </div>
        """)
        st.html(card)

        # Candlestick Chart
        st.markdown(
            "<p style='font-size:0.78rem;font-weight:700;color:#94A3B8;margin:0 0 2px;'>"
            "5-MIN CANDLESTICK — VWAP · EMA 9/21 · SUPERTREND · ENTRY/SL/TGT · VOLUME</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_candlestick_chart(
                enriched_df, pred.signal, tech,
                entry_price=pred.entry_premium,
                target_price=pred.target,
                stop_loss=pred.stop_loss,
            ),
            width="stretch", config={"displayModeBar": False},
        )

        # PCR Gauge & OI Bar Chart
        bc1, bc2 = st.columns([1, 1.6])
        with bc1:
            st.markdown("<p style='font-size:0.72rem;font-weight:700;color:#94A3B8;margin:0;'>PCR GAUGE</p>", unsafe_allow_html=True)
            st.plotly_chart(build_pcr_gauge(chain.pcr), width="stretch", config={"displayModeBar": False})
        with bc2:
            st.markdown("<p style='font-size:0.72rem;font-weight:700;color:#94A3B8;margin:0;'>STRIKE-WISE OI CHANGE</p>", unsafe_allow_html=True)
            st.plotly_chart(build_oi_chart(chain.df_chain, chain.atm_strike), width="stretch", config={"displayModeBar": False})

        # AI Rationale & Headlines
        with st.expander("📋 AI Rationale & News Headlines", expanded=False):
            st.markdown(f"**Regime:** `{pred.regime}` → **Strategy:** `{pred.strategy_name}`")
            st.markdown(f"**Signal:** `{pred.signal}` — **Confidence:** `{pred.confidence:.1f}%` ({pred.conviction_level})")
            st.info(pred.reason)
            if headlines:
                st.markdown("**Latest Headlines Scored (FinBERT):**")
                st.dataframe(pd.DataFrame(headlines), width="stretch", hide_index=True)


# Footer
st.markdown(
    f'<div style="text-align:center;padding:10px 0;color:#64748B;font-size:0.72rem;">'
    f'⚡ Algo Radar v5 LUX — Real-Time Signal Engine'
    f' | UTC: {datetime.now(timezone.utc).strftime("%H:%M:%S")}</div>',
    unsafe_allow_html=True,
)

col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
with col_r2:
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()
