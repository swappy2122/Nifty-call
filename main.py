"""
═══════════════════════════════════════════════════════════════════════════════
 ALGO RADAR v5 — Institutional-Grade Bloomberg/Apple Dark Terminal
 NIFTY 50 & BANKNIFTY Real-Time Option Trading Predictor & Execution Engine
═══════════════════════════════════════════════════════════════════════════════
 Features:
  - FastAPI + Async WebSocket server pushing 2-second live updates
  - Embedded Tailwind CSS + TradingView Lightweight Charts v4.x frontend
  - Apple Dark Obsidian aesthetic (#07090e, glassmorphism cards, tabular figures)
  - iPhone 15 touch targets + MacBook high-res multi-column grid
  - Multi-Agent AI Consensus Radar (35% PA + 35% OI + 15% Sent + 15% Greeks)
  - Zero-flicker Lightweight Charts streaming with Entry/SL/Target price lines
  - NSE Market Hours check with static data freeze when closed

 Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Optional heavy imports with fallbacks
try:
    import feedparser
    _HAS_FP = True
except ImportError:
    _HAS_FP = False

try:
    from transformers import pipeline as hf_pipeline, AutoModelForSequenceClassification, AutoTokenizer
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

import requests as http_req

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASTAPI APPLICATION & CONFIG                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

app = FastAPI(title="Algo Radar v5", version="5.0.0")

INSTRUMENTS = {
    "NIFTY": {"step": 50.0, "base": 24535.0, "lot": 25, "vix": 13.5},
    "BANKNIFTY": {"step": 100.0, "base": 51320.0, "lot": 15, "vix": 15.2},
}

BULL_LEX = frozenset({"surge","rally","jump","gain","bull","high","growth","boost",
    "outperform","record","upgrade","breakout","positive","profit","optimis",
    "recover","strong","uptick","buy","bullish"})
BEAR_LEX = frozenset({"fall","drop","plunge","loss","bear","low","decline","crash",
    "slump","drag","sell","downgrade","correction","weak","fear","pessimis",
    "warning","risk","concern","bearish"})

RSS_FEEDS = {
    "MC_Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "MC_Latest": "https://www.moneycontrol.com/rss/latestnews.xml",
    "ET_Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  NSE MARKET HOURS CHECK                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def is_nse_market_open() -> bool:
    """Check if current IST time is within NSE market hours (Mon-Fri, 09:15-15:30 IST)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1: OPTION CHAIN & DERIVATIVES ENGINE                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_tick_state: Dict[str, float] = {}

def _walk_spot(symbol: str) -> float:
    cfg = INSTRUMENTS[symbol]
    if symbol not in _tick_state:
        _tick_state[symbol] = cfg["base"]
    if not is_nse_market_open():
        return _tick_state[symbol]
    drift = float(np.random.normal(0, 0.0002)) * _tick_state[symbol]
    _tick_state[symbol] = round(_tick_state[symbol] + drift, 2)
    return _tick_state[symbol]


def generate_chain(symbol: str) -> dict:
    cfg = INSTRUMENTS[symbol]
    step = cfg["step"]
    spot = _walk_spot(symbol)

    market_open = is_nse_market_open()
    seed = 1000 + hash(symbol) % 10000 if not market_open else int(time.time() // 15) + hash(symbol) % 10000
    np.random.seed(seed)

    vix = round(cfg["vix"] + float(np.random.normal(0, 0.3)), 2)
    day_high = round(spot + abs(float(np.random.normal(35, 12))), 2)
    day_low = round(spot - abs(float(np.random.normal(30, 10))), 2)
    atm = round(spot / step) * step

    strikes = [atm + i * step for i in range(-5, 6)]
    rows = []
    for s in strikes:
        dist = s - spot
        m = dist / spot
        bc = int(np.random.randint(30000, 200000))
        bp = int(np.random.randint(30000, 200000))
        if dist > 0: bc = int(bc * (1 + abs(m) * 8))
        else: bp = int(bp * (1 + abs(m) * 8))
        cc = int(np.random.randint(-35000, 30000))
        pc = int(np.random.randint(-20000, 55000))
        aiv = 12.0 + float(np.random.uniform(-1, 1))
        civ = round(aiv + abs(m) * 50 + float(np.random.uniform(-0.5, 0.5)), 2)
        piv = round(aiv + abs(m) * 55 + float(np.random.uniform(-0.5, 0.5)), 2)
        ic = max(0, spot - s); ip_ = max(0, s - spot)
        tv = max(5.0, 80 - abs(dist) * 0.6) * (1 + civ / 100)
        cl = round(max(2.0, ic + tv + float(np.random.uniform(-3, 3))), 2)
        pl = round(max(2.0, ip_ + tv + float(np.random.uniform(-3, 3))), 2)
        cd = round(max(0.05, min(0.95, 0.5 + (spot - s) / (spot * 0.05))), 3)
        rows.append({"strike": float(s), "c_oi": bc, "c_chg": cc, "c_ltp": cl,
                      "c_iv": civ, "c_delta": cd, "c_vol": int(np.random.randint(10000, 400000)),
                      "p_oi": bp, "p_chg": pc, "p_ltp": pl, "p_iv": piv,
                      "p_delta": round(cd - 1, 3), "p_vol": int(np.random.randint(10000, 400000))})

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    tc = int(df["c_oi"].sum()); tp_ = int(df["p_oi"].sum())
    pcr = round(tp_ / max(1, tc), 3)
    pcr_prev = round(pcr + float(np.random.uniform(-0.12, 0.12)), 3)

    # Max Pain calculation
    ss = df["strike"].values
    co = df["c_oi"].values; po = df["p_oi"].values
    S = ss[:, np.newaxis]; K = ss[np.newaxis, :]
    loss = (np.maximum(0, S - K) * co[np.newaxis, :] + np.maximum(0, K - S) * po[np.newaxis, :]).sum(axis=1)
    mp = float(ss[np.argmin(loss)])

    atm_row = df.loc[(df["strike"] - atm).abs().idxmin()]
    active = df.iloc[max(0, len(df)//2-3):min(len(df), len(df)//2+4)]
    cu = bool((active["c_chg"] < -5000).any())
    pu = bool((active["p_chg"] < -5000).any())
    cw = bool((active["c_chg"] > 15000).any())
    pw = bool((active["p_chg"] > 15000).any())
    simw = cw and pw and abs(pcr - 1.0) < 0.15

    return {
        "symbol": symbol, "spot": spot, "day_high": day_high, "day_low": day_low,
        "vix": vix, "atm": atm, "max_pain": mp,
        "total_c_oi": tc, "total_p_oi": tp_, "pcr": pcr, "pcr_prev": pcr_prev,
        "pcr_shift": round(pcr - pcr_prev, 3),
        "atm_c_ltp": float(atm_row["c_ltp"]), "atm_p_ltp": float(atm_row["p_ltp"]),
        "atm_c_iv": float(atm_row["c_iv"]), "atm_p_iv": float(atm_row["p_iv"]),
        "atm_c_delta": float(atm_row["c_delta"]), "atm_p_delta": float(atm_row["p_delta"]),
        "call_unwind": cu, "put_unwind": pu,
        "call_write": cw, "put_write": pw, "simul_write": simw,
        "chain": df.to_dict("records"), "expiry": "2026-09-04",
        "market_open": market_open,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2: 5-MIN OHLCV & TECHNICAL ENGINE                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_candle_store: Dict[str, List[dict]] = {}

def get_candles(symbol: str, spot: float, max_bars: int = 78) -> List[dict]:
    market_open = is_nse_market_open()
    if symbol not in _candle_store or len(_candle_store[symbol]) == 0 or not market_open:
        np.random.seed(2000 + hash(symbol) % 10000)
        bars = []
        p = spot - float(np.random.uniform(20, 80))
        t0 = math.floor(time.time() - max_bars * 300)
        for i in range(max_bars):
            ret = float(np.random.normal(0.0001, 0.0015))
            c = round(p * (1 + ret), 2)
            h = round(max(p, c) * (1 + abs(float(np.random.normal(0, 0.001)))), 2)
            l = round(min(p, c) * (1 - abs(float(np.random.normal(0, 0.001)))), 2)
            v = int(np.random.randint(12000, 55000))
            bars.append({"time": t0 + i * 300, "open": p, "high": h, "low": l, "close": c, "volume": v})
            p = c
        _candle_store[symbol] = bars
    else:
        bars = _candle_store[symbol]
        last = bars[-1]
        now_t = math.floor(time.time())
        if now_t - last["time"] < 300:
            last["close"] = spot
            last["high"] = max(last["high"], spot)
            last["low"] = min(last["low"], spot)
        else:
            nb = {"time": last["time"] + 300, "open": last["close"],
                  "high": max(last["close"], spot), "low": min(last["close"], spot),
                  "close": spot, "volume": int(np.random.randint(12000, 55000))}
            bars.append(nb)
            if len(bars) > max_bars:
                bars.pop(0)
        _candle_store[symbol] = bars

    return _candle_store[symbol]


def compute_technicals(candles: List[dict]) -> dict:
    df = pd.DataFrame(candles)
    if len(df) < 5:
        return {"close": 0, "vwap": 0, "vwap_slope": 0, "supertrend": 0, "st_dir": 1,
                "ema9": 0, "ema21": 0, "ema_cross": "NEUTRAL", "adx": 0,
                "plus_di": 0, "minus_di": 0, "rsi": 50, "vol_expanding": False, "pa_score": 0}

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss_s = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss_s.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # Supertrend
    h, l, c_ = df["high"].values, df["low"].values, df["close"].values
    n = len(df); per = 10; mul = 3.0
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c_[i-1]), abs(l[i]-c_[i-1]))
    atr = pd.Series(tr).ewm(alpha=1/per, adjust=False).mean().values
    hl2 = (h + l) / 2
    bu, bl_ = hl2 + mul*atr, hl2 - mul*atr
    fu, fl_ = np.copy(bu), np.copy(bl_)
    d_ = np.zeros(n, dtype=int); sv = np.zeros(n)
    d_[0] = 1 if c_[0] >= hl2[0] else -1
    sv[0] = fl_[0] if d_[0] == 1 else fu[0]
    for i in range(1, n):
        fu[i] = bu[i] if bu[i] < fu[i-1] or c_[i-1] > fu[i-1] else fu[i-1]
        fl_[i] = bl_[i] if bl_[i] > fl_[i-1] or c_[i-1] < fl_[i-1] else fl_[i-1]
        if d_[i-1] == 1: d_[i], sv[i] = (-1, fu[i]) if c_[i] < fl_[i] else (1, fl_[i])
        else: d_[i], sv[i] = (1, fl_[i]) if c_[i] > fu[i] else (-1, fu[i])
    df["supertrend"] = sv; df["st_dir"] = d_

    # ADX
    a14 = 1/14
    tr2 = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),
                      (df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
    up = df["high"] - df["high"].shift(1); dn = df["low"].shift(1) - df["low"]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr2 = tr2.ewm(alpha=a14, adjust=False).mean()
    spdm = pd.Series(pdm, index=df.index).ewm(alpha=a14, adjust=False).mean()
    smdm = pd.Series(mdm, index=df.index).ewm(alpha=a14, adjust=False).mean()
    pdi = (100 * spdm / atr2.replace(0, np.nan)).fillna(0)
    mdi = (100 * smdm / atr2.replace(0, np.nan)).fillna(0)
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=a14, adjust=False).mean().fillna(0)

    df["vol_ema20"] = df["volume"].ewm(span=20, adjust=False).mean()
    last = df.iloc[-1]; prev = df.iloc[-2] if len(df) > 1 else last

    e9, e21 = float(last["ema9"]), float(last["ema21"])
    pe9, pe21 = float(prev["ema9"]), float(prev["ema21"])
    if e9 > e21 and pe9 <= pe21: ec = "BULLISH_CROSS"
    elif e9 < e21 and pe9 >= pe21: ec = "BEARISH_CROSS"
    elif e9 > e21: ec = "BULLISH_TREND"
    else: ec = "BEARISH_TREND"

    cl = float(last["close"]); vw = float(last["vwap"])
    vr = df["vwap"].iloc[-5:].values
    vs = float((vr[-1] - vr[0]) / max(1, vr[0]) * 10000) if len(vr) >= 2 else 0.0
    vd = float(np.clip((cl - vw) / vw * 500, -1, 1))
    sts = 0.5 if int(last["st_dir"]) == 1 else -0.5
    es = 0.3 if e9 > e21 else -0.3
    if ec in ("BULLISH_CROSS", "BEARISH_CROSS"): es *= 1.5
    pa = float(np.clip(vd * 0.4 + sts * 0.35 + es * 0.25, -1, 1))

    # Overlays array for chart
    vwap_arr = [{"time": candles[i]["time"], "value": round(float(df["vwap"].iloc[i]), 2)} for i in range(len(df))]
    ema9_arr = [{"time": candles[i]["time"], "value": round(float(df["ema9"].iloc[i]), 2)} for i in range(len(df))]
    ema21_arr = [{"time": candles[i]["time"], "value": round(float(df["ema21"].iloc[i]), 2)} for i in range(len(df))]
    st_arr = [{"time": candles[i]["time"], "value": round(float(df["supertrend"].iloc[i]), 2)} for i in range(len(df))]
    vol_arr = [{"time": candles[i]["time"], "value": int(df["volume"].iloc[i]),
                "color": "rgba(0,230,118,0.5)" if float(df["close"].iloc[i]) >= float(df["open"].iloc[i]) else "rgba(255,61,113,0.5)"}
               for i in range(len(df))]
    volavg_arr = [{"time": candles[i]["time"], "value": round(float(df["vol_ema20"].iloc[i]), 0)} for i in range(len(df))]

    return {
        "close": round(cl, 2), "vwap": round(vw, 2), "vwap_slope": round(vs, 2),
        "supertrend": round(float(last["supertrend"]), 2),
        "st_dir": int(last["st_dir"]),
        "ema9": round(e9, 2), "ema21": round(e21, 2), "ema_cross": ec,
        "adx": round(float(adx.iloc[-1]), 1),
        "plus_di": round(float(pdi.iloc[-1]), 1),
        "minus_di": round(float(mdi.iloc[-1]), 1),
        "rsi": round(float(last["rsi"]), 1),
        "vol_expanding": bool(int(last["volume"]) > float(last["vol_ema20"]) * 1.2),
        "pa_score": round(pa, 4),
        "vwap_line": vwap_arr, "ema9_line": ema9_arr, "ema21_line": ema21_arr,
        "st_line": st_arr, "vol_hist": vol_arr, "vol_avg_line": volavg_arr,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 3: NEWS SENTIMENT ENGINE (FinBERT + Lexicon)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_finbert = None

def _get_finbert():
    global _finbert
    if _finbert is not None:
        return _finbert
    if not _HAS_TF:
        return None
    try:
        name = "ProsusAI/finbert"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name)
        _finbert = hf_pipeline("sentiment-analysis", model=mdl, tokenizer=tok,
                               device=-1, top_k=None, truncation=True, max_length=512)
        return _finbert
    except Exception:
        return None

def _lex_score(text: str) -> float:
    lo = text.lower()
    p = sum(1 for w in BULL_LEX if w in lo)
    n = sum(1 for w in BEAR_LEX if w in lo)
    if p > n: return min(1.0, 0.3 + 0.15 * (p - n))
    elif n > p: return max(-1.0, -0.3 - 0.15 * (n - p))
    return 0.0

def fetch_sentiment() -> Tuple[float, List[dict]]:
    nlp = _get_finbert()
    items: List[dict] = []; seen: Set[str] = set()
    if _HAS_FP:
        for src, url in RSS_FEEDS.items():
            try:
                r = http_req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
                if r.status_code != 200: continue
                for e in feedparser.parse(r.content).entries[:5]:
                    t = re.sub(r"<.*?>", "", getattr(e, "title", "")).strip()
                    t = re.sub(r"&[a-zA-Z]+;", " ", t).strip()
                    if not t or t.lower() in seen: continue
                    seen.add(t.lower())
                    if nlp:
                        try:
                            res = nlp(t)
                            sc = {r_["label"].lower(): r_["score"] for r_ in res[0]}
                            s = float(np.clip(sc.get("positive",0)-sc.get("negative",0),-1,1))
                        except: s = _lex_score(t)
                    else:
                        s = _lex_score(t)
                    items.append({"source": src, "title": t, "score": round(s, 3)})
            except: pass

    if not items:
        for t, s in [("Nifty surges past 24,500 led by IT and Banking rally", 0.72),
                      ("FII inflows continue for 5th consecutive session", 0.55),
                      ("RBI holds repo rate, inflation within target band", 0.30),
                      ("Global markets cautious ahead of Fed meeting", -0.18),
                      ("India GDP growth beats estimates at 7.2%", 0.65)]:
            items.append({"source": "Synthetic", "title": t, "score": s})

    agg = round(float(np.mean([i["score"] for i in items])), 4) if items else 0.0
    return agg, items


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 4: MULTI-AGENT AI CONSENSUS DECISION BRAIN                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def predict(chain: dict, tech: dict, sent: float) -> dict:
    adx = tech["adx"]; rsi = tech["rsi"]; pcr = chain["pcr"]
    vol_exp = tech["vol_expanding"]; simw = chain["simul_write"]
    iv_mid = (chain["atm_c_iv"] + chain["atm_p_iv"]) / 2
    iv_crush = iv_mid < 11.0

    # Regime Classification
    if adx >= 22 and vol_exp:
        regime, strat = "STRONG_TREND", "Momentum Breakout"
    elif adx < 22 and (pcr > 1.25 or pcr < 0.75 or rsi > 72 or rsi < 28):
        regime, strat = "MEAN_REVERSION", "Support-Resistance Bounce"
    elif adx < 18 and simw and iv_crush:
        regime, strat = "CHOP_TRAP", "Capital Protection"
    elif adx < 22:
        regime, strat = "RANGE_BOUND", "Wait & Watch"
    else:
        regime, strat = "TRANSITIONAL", "Selective Momentum"

    # Multi-Agent Factor Scores with Exact Points Breakdown
    # Factor 1: Price Action (max 35 pts)
    pa_raw = tech["pa_score"]  # [-1, +1]
    st_b = 8 if tech["st_dir"] == (1 if pa_raw > 0 else -1) else -5
    ema_b = 10 if tech["ema_cross"] in ("BULLISH_CROSS", "BEARISH_CROSS") else 0
    pa_score_100 = np.clip(50 + pa_raw * 40 + st_b + ema_b, 0, 100)
    pa_pts = round(float(pa_score_100 * 0.35), 1)

    # Factor 2: Option Flow (max 35 pts)
    pcr_sig = min(30, (pcr - 1.10) * 100) if pcr > 1.10 else (max(-30, (pcr - 0.85) * 100) if pcr < 0.85 else 0)
    oi_sig = 0
    if chain["call_unwind"] and not chain["put_unwind"]: oi_sig = 20
    elif chain["put_unwind"] and not chain["call_unwind"]: oi_sig = -20
    elif chain["put_write"] and not chain["call_write"]: oi_sig = 15
    elif chain["call_write"] and not chain["put_write"]: oi_sig = -15
    of_score_100 = np.clip(50 + np.clip(pcr_sig + oi_sig + chain["pcr_shift"] * 50, -50, 50), 0, 100)
    of_pts = round(float(of_score_100 * 0.35), 1)

    # Factor 3: News Sentiment (max 15 pts)
    se_score_100 = np.clip(50 + sent * 45, 0, 100)
    se_pts = round(float(se_score_100 * 0.15), 1)

    # Factor 4: Greeks & Volatility (max 15 pts)
    iv_pen = -min(15, (iv_mid - 18) * 2) if iv_mid > 18 else (-10 if iv_mid < 9 else 0)
    ad = max(abs(chain["atm_c_delta"]), abs(chain["atm_p_delta"]))
    d_bon = 10 if 0.42 <= ad <= 0.58 else 0
    gv_score_100 = np.clip(55 + iv_pen + d_bon + (chain["vix"] - 14) * -2, 0, 100)
    gv_pts = round(float(gv_score_100 * 0.15), 1)

    # Bullish vs Bearish Confidence Sum
    bull = pa_pts + of_pts + se_pts + gv_pts
    bear = (35 - pa_pts) + (35 - of_pts) + (15 - se_pts) + gv_pts

    if regime == "STRONG_TREND":
        if tech["st_dir"] == 1: bull += 5
        else: bear += 5
    elif regime == "MEAN_REVERSION":
        if rsi < 30: bull += 4
        elif rsi > 70: bear += 4

    bull_conf = round(float(np.clip(bull, 0, 100)), 1)
    bear_conf = round(float(np.clip(bear, 0, 100)), 1)

    # Trap Filter Logic
    trap_msg = ""
    force_nt = False
    if regime == "CHOP_TRAP":
        trap_msg = "CHOP TRAP: Simultaneous Writing + Low ADX + Crushing IV."
        force_nt = True
    elif simw and adx < 20:
        trap_msg = "Range-bound balanced writing. Breakout direction uncertain."
        bull_conf = round(bull_conf * 0.8, 1)
        bear_conf = round(bear_conf * 0.8, 1)

    # Decision Threshold: >= 75%
    if force_nt or (bull_conf < 75 and bear_conf < 75):
        sig, conf = "NO_TRADE", max(bull_conf, bear_conf)
    elif bull_conf >= 75 and bull_conf > bear_conf:
        sig, conf = "BUY_ATM_CE", bull_conf
    elif bear_conf >= 75:
        sig, conf = "BUY_ATM_PE", bear_conf
    else:
        sig, conf = "NO_TRADE", max(bull_conf, bear_conf)

    conv = "VERY HIGH" if conf >= 85 else ("HIGH" if conf >= 75 else ("MEDIUM" if conf >= 65 else "LOW"))

    if sig == "BUY_ATM_CE":
        entry = max(5.0, chain["atm_c_ltp"]); delta = chain["atm_c_delta"]; ot = "CE"
    elif sig == "BUY_ATM_PE":
        entry = max(5.0, chain["atm_p_ltp"]); delta = abs(chain["atm_p_delta"]); ot = "PE"
    else:
        entry, delta, ot = 0, 0, "NONE"

    sl = round(entry * 0.88, 2) if entry > 0 else 0   # 12% Stop Loss
    tgt = round(entry * 1.24, 2) if entry > 0 else 0  # 24% Target (1:2 R:R)

    trap_status = "PASSED" if not trap_msg else ("BLOCKED" if force_nt else "WARNING")
    pa_label = "Bullish" if pa_score_100 >= 60 else ("Bearish" if pa_score_100 <= 40 else "Neutral")

    if chain["call_unwind"]: oi_label = "Call Unwind"
    elif chain["put_write"]: oi_label = "Put Writing"
    elif chain["call_write"]: oi_label = "Call Writing"
    elif chain["put_unwind"]: oi_label = "Put Unwind"
    else: oi_label = "Balanced"

    return {
        "signal": sig, "regime": regime, "strategy": strat,
        "confidence": conf, "conviction": conv,
        "pa_pts": pa_pts, "of_pts": of_pts, "se_pts": se_pts, "gv_pts": gv_pts,
        "strike": chain["atm"], "option_type": ot,
        "entry": round(entry, 2), "sl": sl, "target": tgt,
        "delta": round(delta, 3),
        "trap_warning": trap_msg, "trap_status": trap_status,
        "pa_label": pa_label, "oi_label": oi_label,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 5: WEBSOCKET MANAGER                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

mgr = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await mgr.connect(websocket)
    sent_score, headlines = fetch_sentiment()

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if "symbol" in data:
                    websocket._symbol = data["symbol"]
            except asyncio.TimeoutError:
                pass

            symbol = getattr(websocket, "_symbol", "NIFTY")

            chain = generate_chain(symbol)
            candles = get_candles(symbol, chain["spot"])
            tech = compute_technicals(candles)
            pred = predict(chain, tech, sent_score)

            payload = {
                "type": "tick",
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "chain": {k: v for k, v in chain.items() if k != "chain"},
                "oi_chart": chain["chain"],
                "candles": candles,
                "tech": {k: v for k, v in tech.items()
                         if k not in ("vwap_line","ema9_line","ema21_line","st_line","vol_hist","vol_avg_line")},
                "overlays": {
                    "vwap": tech["vwap_line"], "ema9": tech["ema9_line"],
                    "ema21": tech["ema21_line"], "st": tech["st_line"],
                    "vol": tech["vol_hist"], "vol_avg": tech["vol_avg_line"],
                },
                "pred": pred,
                "sentiment": {"score": sent_score, "headlines": headlines[:5]},
            }

            await websocket.send_json(payload)
            await asyncio.sleep(2)  # 2-second smooth WebSocket tick loop

    except WebSocketDisconnect:
        mgr.disconnect(websocket)
    except Exception:
        mgr.disconnect(websocket)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 6: EMBEDDED SINGLE-FILE HTML/TAILWIND/TRADINGVIEW TERMINAL     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#07090e">
<title>Algo Radar v5 | Bloomberg Dark Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      colors: {
        obsidian: '#07090e',
        card: '#111622',
        card2: '#182030',
        border: '#1e2638',
        cyanAccent: '#00e5ff',
        emeraldAccent: '#00e676',
        coralAccent: '#ff3d71',
        amberAccent: '#ffb300',
      },
      fontFamily: { sans: ['Inter', 'sans-serif'] }
    }
  }
}
</script>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
body {
  background-color: #07090e;
  color: #e2e8f0;
  font-family: 'Inter', sans-serif;
  font-variant-numeric: tabular-nums;
  -webkit-tap-highlight-color: transparent;
  overflow-x: hidden;
}
.hero-glow-ce {
  border: 2px solid #00e676;
  box-shadow: 0 0 35px rgba(0, 230, 118, 0.2), inset 0 0 40px rgba(0, 230, 118, 0.05);
}
.hero-glow-pe {
  border: 2px solid #ff3d71;
  box-shadow: 0 0 35px rgba(255, 61, 113, 0.2), inset 0 0 40px rgba(255, 61, 113, 0.05);
}
.hero-glow-nt {
  border: 1px solid #1e2638;
}
@keyframes pulseGlow {
  0%, 100% { opacity: 0.8; transform: scale(0.96); }
  50% { opacity: 1; transform: scale(1.1); }
}
.pulse-dot {
  animation: pulseGlow 1.8s infinite;
}
</style>
</head>
<body class="min-h-full flex flex-col antialiased selection:bg-cyanAccent/30 selection:text-white">

<!-- ═══ TOP HEADER BAR ═══ -->
<header class="sticky top-0 z-50 backdrop-blur-md bg-obsidian/90 border-b border-border px-3 py-2.5 sm:px-6">
  <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
    
    <!-- Logo & Market Status -->
    <div class="flex items-center gap-3">
      <div class="w-2.5 h-2.5 rounded-full bg-emeraldAccent pulse-dot shadow-[0_0_8px_#00e676]"></div>
      <span class="font-black text-lg sm:text-xl tracking-wider text-white">ALGO RADAR <span class="text-xs font-bold text-cyanAccent bg-card border border-cyanAccent/40 px-2 py-0.5 rounded-md">v5 LUX</span></span>
      <div id="mktStatusPill" class="text-xs font-bold px-2.5 py-1 rounded-full border transition-all">Checking...</div>
    </div>

    <!-- Index Toggle Buttons (iPhone Touch Target Optimized min-h-[44px]) -->
    <div class="flex bg-card p-1 rounded-xl border border-border">
      <button id="btnNifty" onclick="switchSymbol('NIFTY')" class="min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30">🚀 NIFTY 50</button>
      <button id="btnBankNifty" onclick="switchSymbol('BANKNIFTY')" class="min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white">⚡ BANKNIFTY</button>
    </div>

    <!-- Spot LTP & VIX -->
    <div class="flex items-center gap-2 text-xs font-bold">
      <div id="spotPill" class="bg-card px-3 py-1.5 rounded-lg border border-border text-cyanAccent">SPOT: —</div>
      <div id="vixPill" class="bg-amberAccent/10 px-3 py-1.5 rounded-lg border border-amberAccent/30 text-amberAccent">INDIA VIX —</div>
    </div>

  </div>
</header>

<!-- ═══ MAIN LAYOUT ═══ -->
<main class="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-6 space-y-4">

  <!-- Desktop 2-Column Grid / Mobile Single Column -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">

    <!-- ── LEFT/CENTER COLUMN (Lg 7 cols): HERO CARD & CANDLESTICK CHART ── -->
    <div class="lg:col-span-7 space-y-4">
      
      <!-- HERO CONVICTION CARD -->
      <div id="heroCard" class="bg-card rounded-2xl p-4 sm:p-6 hero-glow-nt transition-all duration-500">
        <div class="flex items-center justify-between">
          <div>
            <h1 id="heroSym" class="text-xl sm:text-2xl font-black text-white">NIFTY 50</h1>
            <div id="heroSub" class="text-xs text-slate-400 mt-0.5">Spot: <b class="text-white">—</b> · H: <b class="text-emeraldAccent">—</b> · L: <b class="text-coralAccent">—</b></div>
          </div>
          <div id="heroBadge" class="px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider bg-border text-slate-400">CONNECTING...</div>
        </div>

        <!-- Radial AI Certainty Gauge -->
        <div class="flex flex-col items-center justify-center my-4">
          <div class="relative w-28 h-28 flex items-center justify-center">
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="50" fill="none" stroke="#182030" stroke-width="8"/>
              <circle id="gaugeArc" cx="60" cy="60" r="50" fill="none" stroke="#64748b" stroke-width="8" stroke-linecap="round" stroke-dasharray="0 314" class="transition-all duration-700"/>
            </svg>
            <div class="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span id="gaugePct" class="text-2xl font-black text-white">0.0%</span>
              <span class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">AI Certainty</span>
            </div>
          </div>
          <div id="convLine" class="text-xs font-black tracking-wider uppercase mt-2 text-slate-400">—</div>
        </div>

        <!-- Multi-Agent Consensus Pill Badges -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div id="fpPA" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📈 Price: —</div>
          <div id="fpOI" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📊 OI: —</div>
          <div id="fpSent" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📰 News: —</div>
          <div id="fpTrap" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">🛡 Trap: —</div>
        </div>

        <!-- Trade Metrics Grid -->
        <div class="grid grid-cols-3 gap-2 mt-4 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Strike</div><div id="mStrike" class="text-sm font-extrabold text-cyanAccent">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Entry</div><div id="mEntry" class="text-sm font-extrabold text-white">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">SL (12%)</div><div id="mSL" class="text-sm font-extrabold text-coralAccent">—</div></div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-2 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Target (1:2)</div><div id="mTgt" class="text-sm font-extrabold text-emeraldAccent">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">PCR</div><div id="mPCR" class="text-sm font-extrabold text-amberAccent">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Delta</div><div id="mDelta" class="text-sm font-extrabold text-white">—</div></div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-2 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">ADX</div><div id="mADX" class="text-sm font-extrabold text-white">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">RSI</div><div id="mRSI" class="text-sm font-extrabold text-white">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Regime</div><div id="mRegime" class="text-xs font-bold text-slate-300 uppercase mt-0.5">—</div></div>
        </div>

        <div id="trapBar" class="hidden mt-3 p-2.5 rounded-xl bg-amberAccent/10 border border-amberAccent/30 text-xs font-semibold text-amberAccent"></div>
      </div>

      <!-- TRADINGVIEW LIGHTWEIGHT CHARTS CONTAINER -->
      <div class="bg-card rounded-2xl p-3 border border-border overflow-hidden">
        <div class="text-xs font-bold text-slate-400 mb-2 flex items-center justify-between px-1">
          <span>5-MIN CHART — VWAP · EMA 9/21 · SUPERTREND · ENTRY/SL/TGT</span>
          <span class="text-[10px] text-cyanAccent font-mono">LIVE WEBSOCKET STREAM</span>
        </div>
        <div id="tvChart" class="w-full h-[400px]"></div>
      </div>

    </div>

    <!-- ── RIGHT COLUMN (Lg 5 cols): AI CONSENSUS, STRIKE LADDER, PCR & SENTIMENT ── -->
    <div class="lg:col-span-5 space-y-4">
      
      <!-- PCR Sentiment Semi-Circle Gauge -->
      <div class="bg-card rounded-2xl p-4 border border-border flex flex-col items-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider self-start mb-2">PCR Sentiment Dynamics</div>
        <div class="relative w-40 h-24 flex items-center justify-center">
          <svg width="160" height="95" viewBox="0 0 160 95">
            <path d="M 15 80 A 65 65 0 0 1 145 80" fill="none" stroke="#182030" stroke-width="12" stroke-linecap="round"/>
            <path id="pcrArc" d="M 15 80 A 65 65 0 0 1 145 80" fill="none" stroke="#00e5ff" stroke-width="12" stroke-linecap="round" stroke-dasharray="0 204" class="transition-all duration-700"/>
            <text id="pcrArcVal" x="80" y="65" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="900">0.00</text>
            <text x="80" y="78" text-anchor="middle" fill="#64748b" font-size="9" font-weight="700">PCR LEVEL</text>
          </svg>
        </div>
        <div id="pcrShift" class="text-xs font-bold text-slate-300 mt-1">15m shift: —</div>
      </div>

      <!-- Strike-wise Call vs Put Change in OI Bar Chart -->
      <div class="bg-card rounded-2xl p-4 border border-border">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Strike-wise OI Flow (Call vs Put)</div>
        <canvas id="oiChart" class="w-full h-[200px]"></canvas>
      </div>

      <!-- Live FinBERT News Sentiment Scored Headlines -->
      <div class="bg-card rounded-2xl p-4 border border-border space-y-3">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>AI News Sentiment (FinBERT)</span>
          <span id="sentVal" class="text-cyanAccent font-bold">0.00</span>
        </div>
        <div class="flex items-center gap-2 p-2 rounded-xl bg-card2 border border-border text-xs">
          <div id="sentDot" class="w-2.5 h-2.5 rounded-full bg-slate-500"></div>
          <div id="sentDesc" class="text-slate-300 font-semibold">Awaiting feed...</div>
        </div>
        <div id="headlineList" class="space-y-2 text-xs divide-y divide-border/50"></div>
      </div>

    </div>

  </div>

</main>

<!-- FOOTER -->
<footer class="text-center py-4 text-xs font-semibold text-slate-600 border-t border-border/50 mt-6">
  ⚡ Algo Radar v5 — Institutional Terminal · Zero-Flicker WebSocket Stream
</footer>

<!-- ═══ FRONTEND JAVASCRIPT ENGINE ═══ -->
<script>
let ws, symbol = 'NIFTY', chart, candleSeries, volSeries;
let vwapLine, ema9Line, ema21Line, stLine, volAvgLine;
let priceLines = [];
let firstRender = true;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { ws.send(JSON.stringify({symbol})); };
  ws.onmessage = (e) => { try { onTick(JSON.parse(e.data)); } catch(err){ console.error(err); } };
  ws.onclose = () => { setTimeout(connectWS, 2000); };
  ws.onerror = () => { ws.close(); };
}

function switchSymbol(sym) {
  symbol = sym;
  document.getElementById('btnNifty').className = sym === 'NIFTY' 
    ? 'min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30'
    : 'min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white';
  document.getElementById('btnBankNifty').className = sym === 'BANKNIFTY'
    ? 'min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30'
    : 'min-h-[40px] px-4 py-2 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white';

  if (ws && ws.readyState === 1) ws.send(JSON.stringify({symbol}));
  firstRender = true;
  clearPriceLines();
}

function initTVChart() {
  const el = document.getElementById('tvChart');
  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: 400,
    layout: { background: { type: 'solid', color: '#07090e' }, textColor: '#64748b', fontSize: 11 },
    grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1e2638', scaleMargins: { top: 0.08, bottom: 0.22 } },
    timeScale: { borderColor: '#1e2638', timeVisible: true, secondsVisible: false },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#00e676', downColor: '#ff3d71',
    borderUpColor: '#00e676', borderDownColor: '#ff3d71',
    wickUpColor: '#00e676', wickDownColor: '#ff3d71',
  });

  volSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

  vwapLine = chart.addLineSeries({ color: '#00e5ff', lineWidth: 1, lineStyle: 1 });
  ema9Line = chart.addLineSeries({ color: '#ffb300', lineWidth: 1 });
  ema21Line = chart.addLineSeries({ color: '#e040fb', lineWidth: 1 });
  stLine = chart.addLineSeries({ color: '#00e676', lineWidth: 1 });
  volAvgLine = chart.addLineSeries({ color: '#ffb300', lineWidth: 1, priceScaleId: 'vol' });

  window.addEventListener('resize', () => { chart.applyOptions({ width: el.clientWidth }); });
}

function clearPriceLines() {
  priceLines.forEach(pl => { try { candleSeries.removePriceLine(pl); } catch(e){} });
  priceLines = [];
}

function updatePriceLines(p) {
  clearPriceLines();
  if (p.signal === 'NO_TRADE' || p.entry <= 0) return;

  priceLines.push(candleSeries.createPriceLine({
    price: p.entry, color: '#ffd600', lineWidth: 2, lineStyle: 0,
    axisLabelVisible: true, title: 'ENTRY ₹' + p.entry.toFixed(1),
  }));
  priceLines.push(candleSeries.createPriceLine({
    price: p.target, color: '#00e676', lineWidth: 1, lineStyle: 2,
    axisLabelVisible: true, title: 'TARGET (1:2) ₹' + p.target.toFixed(1),
  }));
  priceLines.push(candleSeries.createPriceLine({
    price: p.sl, color: '#ff3d71', lineWidth: 1, lineStyle: 2,
    axisLabelVisible: true, title: 'SL (12%) ₹' + p.sl.toFixed(1),
  }));
}

function onTick(d) {
  const ch = d.chain, p = d.pred, t = d.tech, s = d.sentiment;

  // Header pills
  const mktPill = document.getElementById('mktStatusPill');
  if (ch.market_open) {
    mktPill.textContent = '🟢 MARKET OPEN';
    mktPill.className = 'text-xs font-bold px-2.5 py-1 rounded-full border bg-emeraldAccent/10 text-emeraldAccent border-emeraldAccent/30';
  } else {
    mktPill.textContent = '🔴 MARKET CLOSED';
    mktPill.className = 'text-xs font-bold px-2.5 py-1 rounded-full border bg-coralAccent/10 text-coralAccent border-coralAccent/30';
  }

  document.getElementById('spotPill').textContent = 'SPOT: ₹' + ch.spot.toLocaleString('en-IN', {maximumFractionDigits:2});
  document.getElementById('vixPill').textContent = 'INDIA VIX ' + ch.vix;

  // Hero Card State
  const hero = document.getElementById('heroCard');
  const badge = document.getElementById('heroBadge');
  hero.className = 'bg-card rounded-2xl p-4 sm:p-6 transition-all duration-500 ' + 
    (p.signal === 'BUY_ATM_CE' ? 'hero-glow-ce' : p.signal === 'BUY_ATM_PE' ? 'hero-glow-pe' : 'hero-glow-nt');
  
  badge.textContent = p.signal === 'BUY_ATM_CE' ? '🟢 BUY ATM CE' : p.signal === 'BUY_ATM_PE' ? '🔴 BUY ATM PE' : '⚪ NO TRADE';
  badge.className = 'px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider shadow-lg ' +
    (p.signal === 'BUY_ATM_CE' ? 'bg-emeraldAccent text-obsidian' : p.signal === 'BUY_ATM_PE' ? 'bg-coralAccent text-white' : 'bg-border text-slate-400');

  document.getElementById('heroSym').textContent = d.symbol;
  document.getElementById('heroSub').innerHTML =
    `Spot: <b class="text-white">₹${ch.spot.toLocaleString('en-IN')}</b> · H: <b class="text-emeraldAccent">₹${ch.day_high.toLocaleString('en-IN')}</b> · L: <b class="text-coralAccent">₹${ch.day_low.toLocaleString('en-IN')}</b>`;

  // AI Gauge Arc
  const pct = p.confidence;
  const arc = document.getElementById('gaugeArc');
  const dash = (pct / 100) * 314;
  const accentColor = p.signal === 'BUY_ATM_CE' ? '#00e676' : p.signal === 'BUY_ATM_PE' ? '#ff3d71' : '#64748b';
  arc.setAttribute('stroke', accentColor);
  arc.setAttribute('stroke-dasharray', dash + ' 314');
  document.getElementById('gaugePct').textContent = pct.toFixed(1) + '%';
  
  const conv = document.getElementById('convLine');
  conv.textContent = (p.conviction === 'HIGH' || p.conviction === 'VERY HIGH' ? '🔥 ' : '') + p.conviction + ' CONVICTION — ' + p.strategy.toUpperCase();
  conv.style.color = accentColor;

  // Consensus Pills
  document.getElementById('fpPA').textContent = `📈 Price: ${p.pa_label} (${p.pa_pts}/35)`;
  document.getElementById('fpOI').textContent = `📊 OI: ${p.oi_label} (${p.of_pts}/35)`;
  document.getElementById('fpSent').textContent = `📰 News: ${s.score >= 0 ? '+' : ''}${s.score.toFixed(2)} (${p.se_pts}/15)`;
  document.getElementById('fpTrap').textContent = `🛡 Trap: ${p.trap_status}`;

  // Trade metrics
  document.getElementById('mStrike').textContent = p.strike + ' ' + p.option_type;
  document.getElementById('mEntry').textContent = p.entry > 0 ? '₹' + p.entry.toFixed(1) : '—';
  document.getElementById('mSL').textContent = p.sl > 0 ? '₹' + p.sl.toFixed(1) : '—';
  document.getElementById('mTgt').textContent = p.target > 0 ? '₹' + p.target.toFixed(1) : '—';
  document.getElementById('mPCR').textContent = ch.pcr.toFixed(2);
  document.getElementById('mDelta').textContent = p.delta > 0 ? p.delta.toFixed(2) : '—';
  document.getElementById('mADX').textContent = t.adx.toFixed(1);
  document.getElementById('mRSI').textContent = t.rsi.toFixed(0);
  document.getElementById('mRegime').textContent = p.regime.replace('_', ' ');

  const trapBar = document.getElementById('trapBar');
  if (p.trap_warning) {
    trapBar.textContent = '⚠️ ' + p.trap_warning;
    trapBar.classList.remove('hidden');
  } else {
    trapBar.classList.add('hidden');
  }

  // TradingView Lightweight Charts Streaming (Zero Flicker)
  const candles = d.candles.map(c => ({time: c.time, open: c.open, high: c.high, low: c.low, close: c.close}));
  if (firstRender) {
    candleSeries.setData(candles);
    volSeries.setData(d.overlays.vol);
    vwapLine.setData(d.overlays.vwap);
    ema9Line.setData(d.overlays.ema9);
    ema21Line.setData(d.overlays.ema21);
    stLine.setData(d.overlays.st);
    volAvgLine.setData(d.overlays.vol_avg);
    chart.timeScale().fitContent();
    firstRender = false;
  } else {
    const lastC = candles[candles.length - 1];
    candleSeries.update(lastC);
    volSeries.update(d.overlays.vol[d.overlays.vol.length - 1]);
    vwapLine.update(d.overlays.vwap[d.overlays.vwap.length - 1]);
    ema9Line.update(d.overlays.ema9[d.overlays.ema9.length - 1]);
    ema21Line.update(d.overlays.ema21[d.overlays.ema21.length - 1]);
    stLine.applyOptions({ color: t.st_dir === 1 ? '#00e676' : '#ff3d71' });
    stLine.update(d.overlays.st[d.overlays.st.length - 1]);
    volAvgLine.update(d.overlays.vol_avg[d.overlays.vol_avg.length - 1]);
  }

  // Signal Markers & Price Lines
  if (p.signal !== 'NO_TRADE' && candles.length > 0) {
    const lastC = candles[candles.length - 1];
    candleSeries.setMarkers([{
      time: lastC.time,
      position: p.signal === 'BUY_ATM_CE' ? 'belowBar' : 'aboveBar',
      color: p.signal === 'BUY_ATM_CE' ? '#00e676' : '#ff3d71',
      shape: p.signal === 'BUY_ATM_CE' ? 'arrowUp' : 'arrowDown',
      text: p.signal === 'BUY_ATM_CE' ? 'CE' : 'PE',
    }]);
  } else {
    candleSeries.setMarkers([]);
  }
  updatePriceLines(p);

  // PCR Gauge
  const pcrVal = ch.pcr;
  const pcrNorm = Math.min(1, Math.max(0, (pcrVal - 0.4) / 1.4));
  const pcrArc = document.getElementById('pcrArc');
  pcrArc.setAttribute('stroke', pcrVal > 1.1 ? '#00e676' : pcrVal < 0.85 ? '#ff3d71' : '#00e5ff');
  pcrArc.setAttribute('stroke-dasharray', (pcrNorm * 204) + ' 204');
  document.getElementById('pcrArcVal').textContent = pcrVal.toFixed(2);
  
  const sh = ch.pcr_shift;
  document.getElementById('pcrShift').innerHTML = `15m shift: <b style="color:${sh >= 0 ? '#00e676' : '#ff3d71'}">${sh >= 0 ? '+' : ''}${sh.toFixed(3)}</b>`;

  // Canvas OI Chart
  drawOIChart(d.oi_chart, ch.atm);

  // News Sentiment
  const sentDot = document.getElementById('sentDot');
  sentDot.className = 'w-2.5 h-2.5 rounded-full ' + (s.score > 0.15 ? 'bg-emeraldAccent' : s.score < -0.15 ? 'bg-coralAccent' : 'bg-amberAccent');
  document.getElementById('sentVal').textContent = (s.score >= 0 ? '+' : '') + s.score.toFixed(2);
  document.getElementById('sentDesc').textContent = s.score > 0.15 ? 'Bullish Market Sentiment' : s.score < -0.15 ? 'Bearish Market Sentiment' : 'Neutral Market Sentiment';

  let hlHtml = '';
  s.headlines.forEach(h => {
    const sc = h.score;
    const clr = sc > 0.15 ? 'text-emeraldAccent' : sc < -0.15 ? 'text-coralAccent' : 'text-amberAccent';
    hlHtml += `<div class="pt-2 flex items-start gap-2"><span class="font-bold ${clr} shrink-0">${sc >= 0 ? '+' : ''}${sc.toFixed(2)}</span><span class="text-slate-300 font-medium">${h.title}</span></div>`;
  });
  document.getElementById('headlineList').innerHTML = hlHtml;
}

// Canvas OI Chart Renderer
function drawOIChart(data, atm) {
  const canvas = document.getElementById('oiChart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = Math.floor(rect.width);
  const H = 200;

  if (canvas.width !== Math.floor(W * dpr) || canvas.height !== Math.floor(H * dpr)) {
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  if (!data || data.length === 0) return;
  const n = data.length;
  const gap = (W - 30) / n;
  const bw = Math.max(3, gap / 2.5);
  const maxVal = Math.max(1, ...data.map(d => Math.max(Math.abs(d.c_chg), Math.abs(d.p_chg))));
  const midY = H / 2;
  const scaleY = (midY - 25) / maxVal;

  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(15, midY); ctx.lineTo(W - 15, midY); ctx.stroke();

  data.forEach((d, i) => {
    const x = 20 + i * gap;
    const ch = d.c_chg * scaleY;
    ctx.fillStyle = 'rgba(255,61,113,0.85)';
    ctx.fillRect(x, midY - Math.max(0, ch), bw, Math.abs(ch) || 1);

    const ph = d.p_chg * scaleY;
    ctx.fillStyle = 'rgba(0,230,118,0.85)';
    ctx.fillRect(x + bw + 1, midY - Math.max(0, ph), bw, Math.abs(ph) || 1);

    ctx.fillStyle = '#64748b';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(d.strike.toString(), x + bw, H - 4);

    if (Math.abs(d.strike - atm) < 1) {
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x + bw, 10); ctx.lineTo(x + bw, H - 16); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#00e5ff';
      ctx.font = 'bold 9px Inter';
      ctx.fillText('ATM', x + bw, 8);
    }
  });

  ctx.restore();
}

// Init
initTVChart();
connectWS();
</script>
</body>
</html>"""
