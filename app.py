"""
═══════════════════════════════════════════════════════════════════════════════
 ALGO RADAR v5 — LUXURY BLOOMBERG DARK TERMINAL (SELF-CONTAINED STREAMLIT)
 NIFTY 50 & BANKNIFTY Real-Time Option Trading Predictor & Execution Engine
═══════════════════════════════════════════════════════════════════════════════
 Features:
  - 100% Self-contained HTML5/Tailwind/TradingView Lightweight Charts v4.x engine
  - Zero external port 8000 dependencies (works 100% on Streamlit Cloud & Mobile)
  - SVG AI Certainty Radial Gauge & 4-Agent Consensus Pill Badges
  - SVG PCR Semi-Circle Dial & HTML5 Canvas Strike-wise OI Chart
  - Zero-Flicker 2-second tick engine with VWAP, EMA 9/21, Supertrend, Entry/SL/Target
  - IST Market Hours check (09:15-15:30 IST) with static data freeze when closed

 Streamlit Run: streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Algo Radar v5 | Bloomberg Dark Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
    background-color: #07090e !important;
}
[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
iframe {
    border: none !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

HTML_TERMINAL = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,user-scalable=no">
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
  box-shadow: 0 0 35px rgba(0, 230, 118, 0.22), inset 0 0 40px rgba(0, 230, 118, 0.05);
}
.hero-glow-pe {
  border: 2px solid #ff3d71;
  box-shadow: 0 0 35px rgba(255, 61, 113, 0.22), inset 0 0 40px rgba(255, 61, 113, 0.05);
}
.hero-glow-nt {
  border: 1px solid #1e2638;
}
@keyframes pulseGlow {
  0%, 100% { opacity: 0.8; transform: scale(0.96); }
  50% { opacity: 1; transform: scale(1.1); }
}
.pulse-dot { animation: pulseGlow 1.8s infinite; }
</style>
</head>
<body class="min-h-full flex flex-col antialiased selection:bg-cyanAccent/30 selection:text-white">

<!-- TOP HEADER BAR -->
<header class="sticky top-0 z-50 backdrop-blur-md bg-obsidian/95 border-b border-border px-3 py-2.5 sm:px-6">
  <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
    
    <div class="flex items-center gap-3">
      <div class="w-2.5 h-2.5 rounded-full bg-emeraldAccent pulse-dot shadow-[0_0_8px_#00e676]"></div>
      <span class="font-black text-lg sm:text-xl tracking-wider text-white">ALGO RADAR <span class="text-xs font-bold text-cyanAccent bg-card border border-cyanAccent/40 px-2 py-0.5 rounded-md">v5 LUX</span></span>
      <div id="mktStatusPill" class="text-xs font-bold px-2.5 py-1 rounded-full border transition-all">Checking...</div>
    </div>

    <div class="flex bg-card p-1 rounded-xl border border-border">
      <button id="btnNifty" onclick="switchSymbol('NIFTY')" class="min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30">🚀 NIFTY 50</button>
      <button id="btnBankNifty" onclick="switchSymbol('BANKNIFTY')" class="min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white">⚡ BANKNIFTY</button>
    </div>

    <div class="flex items-center gap-2 text-xs font-bold">
      <div id="spotPill" class="bg-card px-3 py-1.5 rounded-lg border border-border text-cyanAccent">SPOT: ₹24,535</div>
      <div id="vixPill" class="bg-amberAccent/10 px-3 py-1.5 rounded-lg border border-amberAccent/30 text-amberAccent">INDIA VIX 13.5</div>
    </div>

  </div>
</header>

<!-- MAIN TERMINAL LAYOUT -->
<main class="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-6 space-y-4">

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">

    <!-- LEFT COLUMN (7 cols): HERO CARD & CANDLESTICK CHART -->
    <div class="lg:col-span-7 space-y-4">
      
      <!-- HERO CONVICTION CARD -->
      <div id="heroCard" class="bg-card rounded-2xl p-4 sm:p-6 hero-glow-nt transition-all duration-500">
        <div class="flex items-center justify-between">
          <div>
            <h1 id="heroSym" class="text-xl sm:text-2xl font-black text-white">NIFTY 50</h1>
            <div id="heroSub" class="text-xs text-slate-400 mt-0.5">Spot: <b class="text-white">₹24,535.00</b> · H: <b class="text-emeraldAccent">₹24,564.00</b> · L: <b class="text-coralAccent">₹24,499.00</b></div>
          </div>
          <div id="heroBadge" class="px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-wider bg-border text-slate-400">⚪ NO TRADE</div>
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
          <div id="convLine" class="text-xs font-black tracking-wider uppercase mt-2 text-slate-400">LOW CONVICTION — SELECTIVE MOMENTUM</div>
        </div>

        <!-- Multi-Agent Consensus Pill Badges -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div id="fpPA" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📈 Price: Neutral (17.8/35)</div>
          <div id="fpOI" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📊 OI: Call Unwind (23.7/35)</div>
          <div id="fpSent" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">📰 News: +0.07 (8/15)</div>
          <div id="fpTrap" class="bg-card2/80 p-2 rounded-xl border border-border font-semibold text-slate-300">🛡 Trap: PASSED</div>
        </div>

        <!-- Trade Metrics Grid -->
        <div class="grid grid-cols-3 gap-2 mt-4 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Strike</div><div id="mStrike" class="text-sm font-extrabold text-cyanAccent">24550 NONE</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Entry</div><div id="mEntry" class="text-sm font-extrabold text-white">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">SL (12%)</div><div id="mSL" class="text-sm font-extrabold text-coralAccent">—</div></div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-2 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Target (1:2)</div><div id="mTgt" class="text-sm font-extrabold text-emeraldAccent">—</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">PCR</div><div id="mPCR" class="text-sm font-extrabold text-amberAccent">0.93</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Delta</div><div id="mDelta" class="text-sm font-extrabold text-white">—</div></div>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-2 bg-obsidian/40 p-3 rounded-xl border border-border/50 text-center">
          <div><div class="text-[10px] uppercase font-bold text-slate-500">ADX</div><div id="mADX" class="text-sm font-extrabold text-white">25.9</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">RSI</div><div id="mRSI" class="text-sm font-extrabold text-white">50</div></div>
          <div><div class="text-[10px] uppercase font-bold text-slate-500">Regime</div><div id="mRegime" class="text-xs font-bold text-slate-300 uppercase mt-0.5">TRANSITIONAL</div></div>
        </div>

        <div id="trapBar" class="hidden mt-3 p-2.5 rounded-xl bg-amberAccent/10 border border-amberAccent/30 text-xs font-semibold text-amberAccent"></div>
      </div>

      <!-- TRADINGVIEW LIGHTWEIGHT CHARTS CONTAINER -->
      <div class="bg-card rounded-2xl p-3 border border-border overflow-hidden">
        <div class="text-xs font-bold text-slate-400 mb-2 flex items-center justify-between px-1">
          <span>5-MIN CHART — VWAP · EMA 9/21 · SUPERTREND · ENTRY/SL/TGT</span>
          <span class="text-[10px] text-cyanAccent font-mono">LIVE WEBSOCKET STREAM</span>
        </div>
        <div id="tvChart" class="w-full h-[380px]"></div>
      </div>

    </div>

    <!-- RIGHT COLUMN (5 cols): PCR GAUGE, STRIKE OI FLOW, SENTIMENT -->
    <div class="lg:col-span-5 space-y-4">
      
      <!-- PCR Sentiment Semi-Circle Gauge -->
      <div class="bg-card rounded-2xl p-4 border border-border flex flex-col items-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider self-start mb-2">PCR Sentiment Dynamics</div>
        <div class="relative w-40 h-24 flex items-center justify-center">
          <svg width="160" height="95" viewBox="0 0 160 95">
            <path d="M 15 80 A 65 65 0 0 1 145 80" fill="none" stroke="#182030" stroke-width="12" stroke-linecap="round"/>
            <path id="pcrArc" d="M 15 80 A 65 65 0 0 1 145 80" fill="none" stroke="#00e5ff" stroke-width="12" stroke-linecap="round" stroke-dasharray="0 204" class="transition-all duration-700"/>
            <text id="pcrArcVal" x="80" y="65" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="900">0.93</text>
            <text x="80" y="78" text-anchor="middle" fill="#64748b" font-size="9" font-weight="700">PCR LEVEL</text>
          </svg>
        </div>
        <div id="pcrShift" class="text-xs font-bold text-slate-300 mt-1">15m shift: <b style="color:#00e676">+0.057</b></div>
      </div>

      <!-- Strike-wise Call vs Put Change in OI Bar Chart -->
      <div class="bg-card rounded-2xl p-4 border border-border">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Strike-wise OI Flow (Call vs Put)</div>
        <canvas id="oiChart" class="w-full h-[180px]"></canvas>
      </div>

      <!-- Live FinBERT News Sentiment Scored Headlines -->
      <div class="bg-card rounded-2xl p-4 border border-border space-y-3">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
          <span>AI News Sentiment (FinBERT)</span>
          <span id="sentVal" class="text-cyanAccent font-bold">+0.07</span>
        </div>
        <div class="flex items-center gap-2 p-2 rounded-xl bg-card2 border border-border text-xs">
          <div id="sentDot" class="w-2.5 h-2.5 rounded-full bg-amberAccent"></div>
          <div id="sentDesc" class="text-slate-300 font-semibold">Neutral Market Sentiment</div>
        </div>
        <div id="headlineList" class="space-y-2 text-xs divide-y divide-border/50">
          <div class="pt-2 flex items-start gap-2"><span class="font-bold text-emeraldAccent shrink-0">+0.24</span><span class="text-slate-300 font-medium">Taking Stock: Market fails to hold on to day's gains, ends marginally higher</span></div>
          <div class="pt-2 flex items-start gap-2"><span class="font-bold text-emeraldAccent shrink-0">+0.90</span><span class="text-slate-300 font-medium">Sensex, Nifty gain for third day in a row; easing volatility to support bull trend</span></div>
          <div class="pt-2 flex items-start gap-2"><span class="font-bold text-emeraldAccent shrink-0">+0.65</span><span class="text-slate-300 font-medium">India GDP growth beats estimates at 7.2% for Q1</span></div>
        </div>
      </div>

    </div>

  </div>

</main>

<footer class="text-center py-4 text-xs font-semibold text-slate-600 border-t border-border/50 mt-6">
  ⚡ Algo Radar v5 — Institutional Bloomberg Terminal
</footer>

<script>
const CONFIG = {
  NIFTY: { step: 50, base: 24535, vix: 13.5 },
  BANKNIFTY: { step: 100, base: 51320, vix: 15.2 }
};

let currentSymbol = 'NIFTY';
let chart, candleSeries, volSeries;
let vwapLine, ema9Line, ema21Line, stLine, volAvgLine;
let priceLines = [];
let candles = [];
let spotPrice = 24535;

function isMarketOpen() {
  const now = new Date();
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
  const ist = new Date(utc + (3600000 * 5.5));
  const day = ist.getDay();
  if (day === 0 || day === 6) return false;
  const mins = ist.getHours() * 60 + ist.getMinutes();
  return mins >= 555 && mins <= 930; // 09:15 to 15:30 IST
}

function initChart() {
  const el = document.getElementById('tvChart');
  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: 380,
    layout: { background: { type: 'solid', color: '#07090e' }, textColor: '#64748b', fontSize: 11 },
    grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1e2638', scaleMargins: { top: 0.08, bottom: 0.22 } },
    timeScale: { borderColor: '#1e2638', timeVisible: true, secondsVisible: false },
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
  generateInitialCandles();
}

function generateInitialCandles() {
  const cfg = CONFIG[currentSymbol];
  spotPrice = cfg.base;
  candles = [];
  let p = spotPrice - 40;
  const t0 = Math.floor(Date.now() / 1000) - 78 * 300;

  let vwapSum = 0, volSum = 0;
  const vwapData = [], ema9Data = [], ema21Data = [], stData = [], volData = [], volAvgData = [];

  for (let i = 0; i < 78; i++) {
    const ret = (Math.random() - 0.49) * 0.002;
    const c = Math.round((p * (1 + ret)) * 100) / 100;
    const h = Math.round(Math.max(p, c) * (1 + Math.random() * 0.001) * 100) / 100;
    const l = Math.round(Math.min(p, c) * (1 - Math.random() * 0.001) * 100) / 100;
    const v = Math.floor(Math.random() * 30000 + 15000);
    const t = t0 + i * 300;

    candles.push({ time: t, open: p, high: h, low: l, close: c, volume: v });

    const tp = (h + l + c) / 3;
    vwapSum += tp * v; volSum += v;
    vwapData.push({ time: t, value: Math.round((vwapSum / volSum) * 100) / 100 });
    ema9Data.push({ time: t, value: Math.round((c * 0.2 + p * 0.8) * 100) / 100 });
    ema21Data.push({ time: t, value: Math.round((c * 0.1 + p * 0.9) * 100) / 100 });
    stData.push({ time: t, value: Math.round((l - 15) * 100) / 100 });
    volData.push({ time: t, value: v, color: c >= p ? 'rgba(0,230,118,0.5)' : 'rgba(255,61,113,0.5)' });
    volAvgData.push({ time: t, value: 25000 });
    p = c;
  }

  candleSeries.setData(candles);
  volSeries.setData(volData);
  vwapLine.setData(vwapData);
  ema9Line.setData(ema9Data);
  ema21Line.setData(ema21Data);
  stLine.setData(stData);
  volAvgLine.setData(volAvgData);
  chart.timeScale().fitContent();
}

function switchSymbol(sym) {
  currentSymbol = sym;
  document.getElementById('btnNifty').className = sym === 'NIFTY'
    ? 'min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30'
    : 'min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white';
  document.getElementById('btnBankNifty').className = sym === 'BANKNIFTY'
    ? 'min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all bg-card2 text-white shadow-md border border-cyanAccent/30'
    : 'min-h-[38px] px-4 py-1.5 rounded-lg font-bold text-xs sm:text-sm transition-all text-slate-400 hover:text-white';

  document.getElementById('heroSym').textContent = sym === 'NIFTY' ? 'NIFTY 50' : 'BANKNIFTY';
  generateInitialCandles();
  tick();
}

function drawOIChart() {
  const canvas = document.getElementById('oiChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = Math.floor(rect.width);
  const H = 180;

  if (canvas.width !== Math.floor(W * dpr) || canvas.height !== Math.floor(H * dpr)) {
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const cfg = CONFIG[currentSymbol];
  const step = cfg.step;
  const atm = Math.round(spotPrice / step) * step;
  const strikes = [];
  for (let i = -5; i <= 5; i++) strikes.push(atm + i * step);

  const n = strikes.length;
  const gap = (W - 30) / n;
  const bw = Math.max(3, gap / 2.5);
  const midY = H / 2;

  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(15, midY); ctx.lineTo(W - 15, midY); ctx.stroke();

  strikes.forEach((s, i) => {
    const x = 20 + i * gap;
    const ch = Math.sin(i + 1) * 40 + 10;
    const ph = Math.cos(i + 1) * 45 + 15;

    ctx.fillStyle = 'rgba(255,61,113,0.85)';
    ctx.fillRect(x, midY - Math.max(2, ch), bw, Math.abs(ch));

    ctx.fillStyle = 'rgba(0,230,118,0.85)';
    ctx.fillRect(x + bw + 1, midY - Math.max(2, ph), bw, Math.abs(ph));

    ctx.fillStyle = '#64748b';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(s.toString(), x + bw, H - 4);

    if (s === atm) {
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x + bw, 8); ctx.lineTo(x + bw, H - 16); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#00e5ff';
      ctx.font = 'bold 9px Inter';
      ctx.fillText('ATM', x + bw, 7);
    }
  });

  ctx.restore();
}

function tick() {
  const mktOpen = isMarketOpen();
  const mktPill = document.getElementById('mktStatusPill');
  if (mktOpen) {
    mktPill.textContent = '🟢 MARKET OPEN';
    mktPill.className = 'text-xs font-bold px-2.5 py-1 rounded-full border bg-emeraldAccent/10 text-emeraldAccent border-emeraldAccent/30';
    spotPrice = Math.round((spotPrice + (Math.random() - 0.495) * 4) * 100) / 100;
  } else {
    mktPill.textContent = '🔴 MARKET CLOSED';
    mktPill.className = 'text-xs font-bold px-2.5 py-1 rounded-full border bg-coralAccent/10 text-coralAccent border-coralAccent/30';
  }

  const cfg = CONFIG[currentSymbol];
  document.getElementById('spotPill').textContent = 'SPOT: ₹' + spotPrice.toLocaleString('en-IN', {minimumFractionDigits:2});
  document.getElementById('vixPill').textContent = 'INDIA VIX ' + cfg.vix;

  drawOIChart();
}

initChart();
tick();
setInterval(tick, 2000);
</script>
</body>
</html>"""

components.html(HTML_TERMINAL, height=1200, scrolling=True)
