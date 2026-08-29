"""
Sentiment and Technical Momentum Engine
=======================================
Combines real-time financial news sentiment (Moneycontrol & Economic Times via feedparser
and HuggingFace 'ProsusAI/finbert') with 5-minute OHLCV technical indicators
(VWAP, Supertrend, 9 & 21 EMA Crossover, ADX).

Outputs a unified Technical + Sentiment Momentum Score in [-1.0, +1.0] with trade signals.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

try:
    import feedparser
except ImportError:
    feedparser = None

# Configure logging
logger = logging.getLogger("sentiment_technical")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class NewsHeadline:
    """Represents a single scraped financial news item."""
    source: str
    title: str
    summary: str
    published: str
    link: str
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None  # Raw FinBERT confidence [0, 1]
    polarity_score: Optional[float] = None   # Normalized score [-1.0, +1.0]


@dataclass
class SentimentReport:
    """Aggregated sentiment analysis results."""
    aggregate_score: float  # [-1.0, +1.0]
    total_headlines: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    sentiment_label: Literal["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"]
    headlines: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TechnicalIndicators:
    """Latest computed technical metrics for 5-min OHLCV."""
    close: float
    vwap: float
    supertrend: float
    supertrend_direction: int  # +1 (Bullish) or -1 (Bearish)
    ema_9: float
    ema_21: float
    ema_crossover: Literal["BULLISH_CROSS", "BEARISH_CROSS", "BULLISH_TREND", "BEARISH_TREND"]
    adx: float
    plus_di: float
    minus_di: float
    technical_score: float  # [-1.0, +1.0]


@dataclass
class CombinedMomentumResult:
    """Final unified momentum decision metrics."""
    symbol: str
    combined_score: float  # [-1.0, +1.0]
    signal: Literal["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
    technical_score: float
    sentiment_score: float
    technical_details: Dict[str, Any]
    sentiment_details: Dict[str, Any]
    weights: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# News Scraper & FinBERT Sentiment Engine
# ============================================================================

class NewsSentimentAnalyzer:
    """
    Scrapes RSS news feeds from Moneycontrol & Economic Times,
    and runs HuggingFace 'ProsusAI/finbert' to score financial sentiment.
    """

    RSS_FEEDS = {
        "Moneycontrol_Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
        "Moneycontrol_Latest": "https://www.moneycontrol.com/rss/latestnews.xml",
        "Moneycontrol_Business": "https://www.moneycontrol.com/rss/business.xml",
        "EconomicTimes_Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "EconomicTimes_Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    }

    def __init__(self, device: Optional[int] = -1, use_pipeline: bool = True) -> None:
        """
        Args:
            device: Device for transformer pipeline (-1 for CPU, 0 for CUDA/MPS).
            use_pipeline: Whether to initialize transformer pipeline immediately.
        """
        self.device = device
        self._pipeline = None
        self._use_pipeline = use_pipeline

    def _get_pipeline(self):
        """Lazy load HuggingFace FinBERT pipeline."""
        if self._pipeline is None and self._use_pipeline:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                model_name = "ProsusAI/finbert"
                logger.info("Loading FinBERT model: %s...", model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT pipeline loaded successfully.")
            except Exception as e:
                logger.error("Failed to load FinBERT pipeline: %s. Falling back to heuristic scoring.", e)
                self._pipeline = None
        return self._pipeline

    def clean_text(self, text: str) -> str:
        """Remove HTML tags, excess spaces, and clean text."""
        if not text:
            return ""
        clean = re.sub(r"<.*?>", "", text)
        clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def fetch_latest_news(self, max_per_feed: int = 10) -> List[NewsHeadline]:
        """
        Scrape latest headlines from Moneycontrol and Economic Times RSS feeds.

        Args:
            max_per_feed: Maximum headlines to fetch per RSS source.

        Returns:
            List of NewsHeadline instances.
        """
        headlines: List[NewsHeadline] = []
        seen_titles = set()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }

        for source_name, url in self.RSS_FEEDS.items():
            try:
                logger.info("Fetching RSS feed from %s...", source_name)
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    logger.warning("Feed %s returned status %d", source_name, resp.status_code)
                    continue

                if feedparser is not None:
                    feed = feedparser.parse(resp.content)
                    entries = feed.entries[:max_per_feed]
                else:
                    # Basic XML parsing fallback
                    entries = []

                for entry in entries:
                    raw_title = getattr(entry, "title", "")
                    title = self.clean_text(raw_title)
                    if not title or title.lower() in seen_titles:
                        continue

                    seen_titles.add(title.lower())
                    summary = self.clean_text(getattr(entry, "summary", ""))
                    published = getattr(entry, "published", str(datetime.now(timezone.utc)))
                    link = getattr(entry, "link", "")

                    headlines.append(
                        NewsHeadline(
                            source=source_name,
                            title=title,
                            summary=summary,
                            published=published,
                            link=link,
                        )
                    )
            except Exception as e:
                logger.warning("Error fetching RSS feed %s: %s", source_name, e)

        logger.info("Scraped %d unique headlines across feeds.", len(headlines))
        return headlines

    def score_headline(self, text: str) -> Tuple[str, float, float]:
        """
        Score a single headline using FinBERT into polarity in [-1.0, +1.0].

        Returns:
            Tuple of (label, raw_confidence, normalized_polarity)
        """
        nlp = self._get_pipeline()
        if nlp is not None:
            try:
                # FinBERT returns a list of dicts: [{'label': 'positive', 'score': 0.95}, ...]
                res = nlp(text)
                scores = {item["label"].lower(): item["score"] for item in res[0]}
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                neu = scores.get("neutral", 0.0)

                # Polarity score: pos - neg (neutral dampens polarity)
                polarity = float(np.clip(pos - neg, -1.0, 1.0))
                best_label = max(scores.items(), key=lambda x: x[1])[0]
                raw_score = scores[best_label]
                return best_label.upper(), raw_score, round(polarity, 4)
            except Exception as e:
                logger.warning("FinBERT inference error on '%s': %s", text[:40], e)

        # Heuristic Lexicon Fallback
        lower = text.lower()
        bull_words = {"surge", "jump", "rally", "gain", "profit", "bull", "high", "growth", "boost", "outperform"}
        bear_words = {"fall", "drop", "plunge", "loss", "bear", "low", "decline", "crash", "slump", "drag", "down"}

        pos_count = sum(1 for w in bull_words if w in lower)
        neg_count = sum(1 for w in bear_words if w in lower)

        if pos_count > neg_count:
            return "POSITIVE", 0.75, round(min(1.0, 0.4 + 0.2 * (pos_count - neg_count)), 4)
        elif neg_count > pos_count:
            return "NEGATIVE", 0.75, round(max(-1.0, -0.4 - 0.2 * (neg_count - pos_count)), 4)
        else:
            return "NEUTRAL", 0.50, 0.0

    def analyze_sentiment(self, headlines: Optional[List[NewsHeadline]] = None) -> SentimentReport:
        """
        Analyze news sentiment across all scraped or supplied headlines.

        Args:
            headlines: Optional pre-scraped headlines. If None, scrapes live.

        Returns:
            Structured SentimentReport.
        """
        if headlines is None:
            headlines = self.fetch_latest_news()

        if not headlines:
            logger.warning("No headlines available to analyze.")
            return SentimentReport(
                aggregate_score=0.0,
                total_headlines=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                sentiment_label="NEUTRAL",
            )

        scored_data = []
        polarities = []
        bullish = 0
        bearish = 0
        neutral = 0

        for item in headlines:
            label, conf, polarity = self.score_headline(item.title)
            item.sentiment_label = label
            item.sentiment_score = conf
            item.polarity_score = polarity

            polarities.append(polarity)
            if polarity > 0.15:
                bullish += 1
            elif polarity < -0.15:
                bearish += 1
            else:
                neutral += 1

            scored_data.append({
                "source": item.source,
                "title": item.title,
                "sentiment_label": label,
                "confidence": round(conf, 4),
                "polarity": polarity,
                "link": item.link,
            })

        # Calculate recency/mean aggregate score
        aggregate_score = float(np.clip(np.mean(polarities), -1.0, 1.0))

        if aggregate_score >= 0.35:
            sentiment_label = "STRONG_BULLISH"
        elif aggregate_score >= 0.10:
            sentiment_label = "BULLISH"
        elif aggregate_score <= -0.35:
            sentiment_label = "STRONG_BEARISH"
        elif aggregate_score <= -0.10:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        return SentimentReport(
            aggregate_score=round(aggregate_score, 4),
            total_headlines=len(headlines),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            sentiment_label=sentiment_label,
            headlines=scored_data,
        )


# ============================================================================
# Technical Analysis Engine (5-min OHLCV)
# ============================================================================

class TechnicalAnalysisEngine:
    """
    Computes technical indicators (VWAP, Supertrend, 9 & 21 EMA Crossover, ADX)
    and evaluates Technical Momentum score in [-1.0, +1.0].
    """

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        Typical Price = (High + Low + Close) / 3
        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = typical_price * df["volume"]
        vwap = pv.cumsum() / df["volume"].cumsum()
        return vwap.ffill().bfill()

    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        """Compute Exponential Moving Average."""
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_supertrend(
        df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute Supertrend Indicator & Direction.

        Returns:
            Tuple of (supertrend_line, supertrend_direction)
            Direction is +1 (Bullish) or -1 (Bearish).
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        # Calculate True Range (TR)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Average True Range (ATR) with Wilder's Smoothing
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = np.copy(basic_upper)
        final_lower = np.copy(basic_lower)
        supertrend = np.zeros(n)
        direction = np.zeros(n, dtype=int)

        # Initial conditions
        direction[0] = 1 if close[0] >= hl2[0] else -1
        supertrend[0] = final_lower[0] if direction[0] == 1 else final_upper[0]

        for i in range(1, n):
            # Final Upper Band
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Final Lower Band
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # Direction & Supertrend Line
            if direction[i - 1] == 1:
                if close[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
            else:
                if close[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]

        return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX), +DI, and -DI using Wilder's Smoothing.

        Returns:
            Tuple of (ADX, Plus_DI, Minus_DI)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's Smoothing
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        smooth_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        smooth_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * (smooth_plus_dm / atr.replace(0, np.nan))
        minus_di = 100.0 * (smooth_minus_dm / atr.replace(0, np.nan))

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)

    def analyze_dataframe(self, ohlcv_df: pd.DataFrame) -> Tuple[pd.DataFrame, TechnicalIndicators]:
        """
        Calculate all indicators across DataFrame and extract latest technical state.

        Args:
            ohlcv_df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

        Returns:
            Tuple of (enriched_df, latest_indicators)
        """
        df = ohlcv_df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Required OHLCV column '{col}' is missing.")
            df[col] = df[col].astype(float)

        # Compute indicators
        df["vwap"] = self.calculate_vwap(df)
        df["ema_9"] = self.calculate_ema(df["close"], span=9)
        df["ema_21"] = self.calculate_ema(df["close"], span=21)
        df["supertrend"], df["supertrend_dir"] = self.calculate_supertrend(df, period=10, multiplier=3.0)
        df["adx"], df["plus_di"], df["minus_di"] = self.calculate_adx(df, period=14)

        # Latest values
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row

        close = float(last_row["close"])
        vwap = float(last_row["vwap"])
        st_dir = int(last_row["supertrend_dir"])
        st_line = float(last_row["supertrend"])
        ema9 = float(last_row["ema_9"])
        ema21 = float(last_row["ema_21"])
        prev_ema9 = float(prev_row["ema_9"])
        prev_ema21 = float(prev_row["ema_21"])
        adx = float(last_row["adx"])
        plus_di = float(last_row["plus_di"])
        minus_di = float(last_row["minus_di"])

        # Determine EMA Crossover state
        if ema9 > ema21 and prev_ema9 <= prev_ema21:
            ema_crossover = "BULLISH_CROSS"
        elif ema9 < ema21 and prev_ema9 >= prev_ema21:
            ema_crossover = "BEARISH_CROSS"
        elif ema9 > ema21:
            ema_crossover = "BULLISH_TREND"
        else:
            ema_crossover = "BEARISH_TREND"

        # Calculate Component Technical Scores (each contributing up to 0.25)
        # 1. VWAP Component (Price > VWAP is bullish)
        vwap_diff_pct = (close - vwap) / vwap
        vwap_score = np.clip(vwap_diff_pct * 100, -0.25, 0.25)

        # 2. Supertrend Component (+0.25 for Bullish, -0.25 for Bearish)
        st_score = 0.25 if st_dir == 1 else -0.25

        # 3. EMA 9/21 Component
        ema_diff_pct = (ema9 - ema21) / ema21
        ema_score = np.clip(ema_diff_pct * 100, -0.25, 0.25)
        if ema_crossover == "BULLISH_CROSS":
            ema_score = 0.25
        elif ema_crossover == "BEARISH_CROSS":
            ema_score = -0.25

        # 4. ADX & Directional Index Component
        # If ADX > 25, trend is strong; polarity determined by plus_di vs minus_di
        di_diff = plus_di - minus_di
        adx_strength = min(1.0, adx / 40.0)  # Max out at 40 ADX
        adx_direction = np.sign(di_diff)
        adx_score = np.clip((adx_strength * adx_direction) * 0.25, -0.25, 0.25)

        # Total Technical Score [-1.0, +1.0]
        total_tech_score = float(np.clip(vwap_score + st_score + ema_score + adx_score, -1.0, 1.0))

        indicators = TechnicalIndicators(
            close=close,
            vwap=round(vwap, 2),
            supertrend=round(st_line, 2),
            supertrend_direction=st_dir,
            ema_9=round(ema9, 2),
            ema_21=round(ema21, 2),
            ema_crossover=ema_crossover,
            adx=round(adx, 2),
            plus_di=round(plus_di, 2),
            minus_di=round(minus_di, 2),
            technical_score=round(total_tech_score, 4),
        )

        return df, indicators


# ============================================================================
# Combined Momentum Pipeline
# ============================================================================

class CombinedMomentumPipeline:
    """
    Integrates FinBERT News Sentiment Analyzer with Technical Indicator Engine
    to generate a combined trade signal and momentum rating.
    """

    def __init__(
        self,
        tech_weight: float = 0.65,
        sent_weight: float = 0.35,
        news_analyzer: Optional[NewsSentimentAnalyzer] = None,
        tech_engine: Optional[TechnicalAnalysisEngine] = None,
    ) -> None:
        """
        Args:
            tech_weight: Weight for technical indicators (default 0.65).
            sent_weight: Weight for sentiment analysis (default 0.35).
            news_analyzer: Custom NewsSentimentAnalyzer instance.
            tech_engine: Custom TechnicalAnalysisEngine instance.
        """
        total = tech_weight + sent_weight
        self.tech_weight = tech_weight / total
        self.sent_weight = sent_weight / total
        self.news_analyzer = news_analyzer or NewsSentimentAnalyzer()
        self.tech_engine = tech_engine or TechnicalAnalysisEngine()

    def evaluate(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        headlines: Optional[List[NewsHeadline]] = None,
    ) -> CombinedMomentumResult:
        """
        Compute Combined Momentum Score and generate signal.

        Args:
            symbol: Trading ticker (e.g., 'NIFTY', 'RELIANCE').
            ohlcv_df: 5-min OHLCV Pandas DataFrame.
            headlines: Optional pre-fetched news headlines.

        Returns:
            CombinedMomentumResult instance.
        """
        # 1. Technical Analysis
        _, tech_metrics = self.tech_engine.analyze_dataframe(ohlcv_df)

        # 2. News Sentiment Analysis
        sent_report = self.news_analyzer.analyze_sentiment(headlines)

        # 3. Weighted Combined Score [-1.0, +1.0]
        combined_score = (self.tech_weight * tech_metrics.technical_score) + (
            self.sent_weight * sent_report.aggregate_score
        )
        combined_score = float(np.clip(combined_score, -1.0, 1.0))

        # 4. Generate Signal
        if combined_score >= 0.45:
            signal = "STRONG_BUY"
        elif combined_score >= 0.15:
            signal = "BUY"
        elif combined_score <= -0.45:
            signal = "STRONG_SELL"
        elif combined_score <= -0.15:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return CombinedMomentumResult(
            symbol=symbol.upper(),
            combined_score=round(combined_score, 4),
            signal=signal,
            technical_score=tech_metrics.technical_score,
            sentiment_score=sent_report.aggregate_score,
            technical_details=asdict(tech_metrics),
            sentiment_details=asdict(sent_report),
            weights={"technical": self.tech_weight, "sentiment": self.sent_weight},
        )


# ============================================================================
# Synthetic Data Generator & CLI Verification
# ============================================================================

def generate_mock_ohlcv(bars: int = 100, base_price: float = 24500.0) -> pd.DataFrame:
    """Generate realistic 5-minute synthetic OHLCV candle data."""
    np.random.seed(42)
    returns = np.random.normal(loc=0.0002, scale=0.002, size=bars)
    prices = base_price * np.cumprod(1 + returns)

    highs = prices * (1 + np.abs(np.random.normal(0, 0.001, size=bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.001, size=bars)))
    opens = np.roll(prices, 1)
    opens[0] = base_price
    volumes = np.random.randint(5000, 50000, size=bars)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })
    return df


if __name__ == "__main__":
    print("=" * 75)
    print("SENTIMENT & TECHNICAL MOMENTUM ENGINE - DEMO EXECUTION")
    print("=" * 75)

    # 1. Scrape and analyze news
    news_analyzer = NewsSentimentAnalyzer(use_pipeline=True)
    headlines = news_analyzer.fetch_latest_news(max_per_feed=3)

    if not headlines:
        print("No live RSS available; utilizing sample headlines for demo.")
        headlines = [
            NewsHeadline(
                source="Moneycontrol",
                title="Markets hit record high as Nifty surges past 24,500 led by IT and Banking rally",
                summary="Robust FII buying and positive earnings lift indices.",
                published="Now",
                link="https://moneycontrol.com",
            ),
            NewsHeadline(
                source="EconomicTimes",
                title="Retail inflation drops to multi-month low, boosting rate cut expectations",
                summary="RBI may consider stance change.",
                published="Now",
                link="https://economictimes.com",
            ),
        ]

    # 2. Generate 5-min OHLCV
    ohlcv = generate_mock_ohlcv(bars=80, base_price=24500.0)

    # 3. Run Combined Pipeline
    pipeline = CombinedMomentumPipeline(tech_weight=0.65, sent_weight=0.35, news_analyzer=news_analyzer)
    result = pipeline.evaluate(symbol="NIFTY", ohlcv_df=ohlcv, headlines=headlines)

    print(f"\n[Symbol]: {result.symbol}")
    print(f"[Combined Momentum Score]: {result.combined_score:+.4f} (Scale: -1.0 to +1.0)")
    print(f"[Signal]: {result.signal}")
    print(f"[Technical Score]: {result.technical_score:+.4f}")
    print(f"  - Close: {result.technical_details['close']:.2f}")
    print(f"  - VWAP: {result.technical_details['vwap']:.2f}")
    print(f"  - Supertrend: {result.technical_details['supertrend']:.2f} (Dir: {result.technical_details['supertrend_direction']})")
    print(f"  - EMA 9/21: {result.technical_details['ema_9']:.2f} / {result.technical_details['ema_21']:.2f} ({result.technical_details['ema_crossover']})")
    print(f"  - ADX: {result.technical_details['adx']:.2f} (Plus DI: {result.technical_details['plus_di']:.2f}, Minus DI: {result.technical_details['minus_di']:.2f})")
    print(f"\n[Sentiment Score]: {result.sentiment_score:+.4f} ({result.sentiment_details['sentiment_label']})")
    print(f"  - Total Headlines: {result.sentiment_details['total_headlines']}")
    print(f"  - Bullish / Bearish / Neutral: {result.sentiment_details['bullish_count']} / {result.sentiment_details['bearish_count']} / {result.sentiment_details['neutral_count']}")
    print("=" * 75)
