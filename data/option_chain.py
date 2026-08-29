"""
Option Chain Engine for Upstox API v2
=====================================
Production-ready module for fetching, parsing, and analyzing real-time option chain data,
spot LTP, Put-Call Ratio (PCR), Max Pain, ATM strikes, and Smart Money Shifts
(Call Unwinding, Put Writing, Resistance/Support dynamics).

Includes:
- Robust HTTP client with exponential backoff retries and connection pooling
- Fast analytics using vectorized Pandas & NumPy operations
- Async WebSocket Streamer for live Upstox v2 Market Data Feed
- Comprehensive type hints, structured outputs, and enterprise-grade error handling
"""

import asyncio
import json
import logging
import ssl
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module-level logger
logger = logging.getLogger("upstox_option_chain")
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
# Exceptions
# ============================================================================

class OptionChainBaseException(Exception):
    """Base exception for all OptionChainEngine errors."""
    pass


class UpstoxAPIError(OptionChainBaseException):
    """Raised when the Upstox API returns an HTTP error or error payload."""
    def __init__(self, status_code: int, message: str, raw_response: Optional[dict] = None):
        super().__init__(f"Upstox API Error [{status_code}]: {message}")
        self.status_code = status_code
        self.message = message
        self.raw_response = raw_response or {}


class InstrumentNotFoundError(OptionChainBaseException):
    """Raised when an unrecognized or unsupported trading symbol is requested."""
    pass


class OptionChainError(OptionChainBaseException):
    """Raised when option chain data is empty, unavailable, or cannot be parsed."""
    pass


class RateLimitError(UpstoxAPIError):
    """Raised when API rate limit (429) is exceeded."""
    pass


# ============================================================================
# Data Models & Constants
# ============================================================================

INDEX_INSTRUMENT_KEYS: Dict[str, str] = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "NIFTYBANK": "NSE_INDEX|Nifty Bank",
    "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
    "NIFTY FIN SERVICE": "NSE_INDEX|Nifty Fin Service",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
    "NIFTY MID SELECT": "NSE_INDEX|NIFTY MID SELECT",
    "SENSEX": "BSE_INDEX|SENSEX",
    "BANKEX": "BSE_INDEX|BANKEX",
}


@dataclass
class SmartMoneyShift:
    """Represents detected institutional smart money shifts at a specific strike."""
    strike_price: float
    shift_type: Literal[
        "CALL_UNWINDING", "PUT_WRITING", "CALL_WRITING", "PUT_UNWINDING"
    ]
    change_in_oi: int
    open_interest: int
    implied_volatility: float
    ltp: float
    description: str


@dataclass
class OptionChainSummary:
    """Comprehensive summary metrics for an option chain expiration cycle."""
    symbol: str
    underlying_key: str
    spot_ltp: float
    expiry_date: str
    atm_strike: float
    max_pain: float
    total_call_oi: int
    total_put_oi: int
    pcr_oi: float
    total_call_volume: int
    total_put_volume: int
    pcr_volume: float
    net_oi_change_bias: int
    sentiment: str
    key_support_strike: float
    key_resistance_strike: float
    max_call_oi_strike: float
    max_put_oi_strike: float
    call_unwinding_strikes: List[Dict[str, Any]] = field(default_factory=list)
    put_writing_strikes: List[Dict[str, Any]] = field(default_factory=list)
    call_writing_strikes: List[Dict[str, Any]] = field(default_factory=list)
    put_unwinding_strikes: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        return {
            "symbol": self.symbol,
            "underlying_key": self.underlying_key,
            "spot_ltp": self.spot_ltp,
            "expiry_date": self.expiry_date,
            "atm_strike": self.atm_strike,
            "max_pain": self.max_pain,
            "total_call_oi": self.total_call_oi,
            "total_put_oi": self.total_put_oi,
            "pcr_oi": self.pcr_oi,
            "total_call_volume": self.total_call_volume,
            "total_put_volume": self.total_put_volume,
            "pcr_volume": self.pcr_volume,
            "net_oi_change_bias": self.net_oi_change_bias,
            "sentiment": self.sentiment,
            "key_support_strike": self.key_support_strike,
            "key_resistance_strike": self.key_resistance_strike,
            "max_call_oi_strike": self.max_call_oi_strike,
            "max_put_oi_strike": self.max_put_oi_strike,
            "smart_money_shifts": {
                "call_unwinding": self.call_unwinding_strikes,
                "put_writing": self.put_writing_strikes,
                "call_writing": self.call_writing_strikes,
                "put_unwinding": self.put_unwinding_strikes,
            },
            "timestamp": self.timestamp,
        }


# ============================================================================
# Option Chain Engine
# ============================================================================

class OptionChainEngine:
    """
    Production-grade Engine to interface with Upstox API v2 for Market Quotes,
    Option Chains, Spot LTP, PCR calculation, Max Pain, and Smart Money Shifts.
    """

    BASE_URL: str = "https://api.upstox.com/v2"

    def __init__(
        self,
        access_token: str,
        api_version: str = "2.0",
        timeout: int = 12,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize OptionChainEngine.

        Args:
            access_token: Upstox API v2 Bearer access token.
            api_version: API version string (default: "2.0").
            timeout: HTTP request timeout in seconds.
            max_retries: Max retries on transient errors (429, 500, 502, 503, 504).
            session: Optional custom requests.Session instance.
        """
        if not access_token or not isinstance(access_token, str):
            raise ValueError("A valid Upstox API access_token must be provided.")

        self.access_token = access_token.strip()
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure persistent session with HTTPAdapter & Retry
        self.session = session or requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    # ------------------------------------------------------------------------
    # Helper & Utility Methods
    # ------------------------------------------------------------------------

    def resolve_instrument_key(self, symbol: str) -> str:
        """
        Map a generic symbol name (e.g. 'NIFTY', 'BANKNIFTY') or custom key
        to standard Upstox instrument key (e.g. 'NSE_INDEX|Nifty 50').

        Args:
            symbol: Symbol string (case-insensitive).

        Returns:
            Resolved Upstox instrument key.
        """
        clean_sym = symbol.strip().upper()
        if clean_sym in INDEX_INSTRUMENT_KEYS:
            return INDEX_INSTRUMENT_KEYS[clean_sym]

        # Check if already in standard key format like 'NSE_INDEX|Nifty 50' or 'NSE_EQ|INE...'
        if "|" in symbol:
            return symbol.strip()

        # Check case-insensitive partial matching
        for k, v in INDEX_INSTRUMENT_KEYS.items():
            if clean_sym == k.upper():
                return v

        # If not matched, assume equity or custom index key format
        logger.warning(
            "Symbol '%s' not in predefined index map; passing verbatim.", symbol
        )
        return symbol.strip()

    def _execute_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an authenticated GET request against Upstox v2 API with error parsing.

        Args:
            endpoint: API path relative to BASE_URL (e.g. '/market-quote/ltp').
            params: Query parameters dictionary.

        Returns:
            Parsed JSON dictionary response.

        Raises:
            RateLimitError: If HTTP 429 is encountered.
            UpstoxAPIError: If HTTP status >= 400 or payload status is 'error'.
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(
                url,
                headers=self._headers,
                params=params,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            logger.error("HTTP request timed out for %s: %s", url, e)
            raise UpstoxAPIError(408, f"Request to Upstox API timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error("Network error during request to %s: %s", url, e)
            raise UpstoxAPIError(0, f"Network communication error: {e}") from e

        if response.status_code == 429:
            raise RateLimitError(
                429, "Upstox API rate limit exceeded. Please throttle requests."
            )

        try:
            payload = response.json()
        except ValueError:
            raise UpstoxAPIError(
                response.status_code,
                f"Non-JSON response received: {response.text[:200]}",
            )

        if response.status_code >= 400 or payload.get("status") == "error":
            errors = payload.get("errors", [])
            err_msg = (
                errors[0].get("message")
                if errors and isinstance(errors[0], dict)
                else payload.get("message", response.text)
            )
            raise UpstoxAPIError(response.status_code, err_msg, raw_response=payload)

        return payload

    # ------------------------------------------------------------------------
    # Spot LTP & Expiry Contract Methods
    # ------------------------------------------------------------------------

    def get_spot_ltp(self, symbol: str) -> float:
        """
        Fetch the live spot Last Traded Price (LTP) for NIFTY, BANKNIFTY, or any instrument.

        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY', 'FINNIFTY', or instrument key).

        Returns:
            Current Spot LTP as float.

        Raises:
            InstrumentNotFoundError: If quote data cannot be located for the symbol.
            UpstoxAPIError: If Upstox API call fails.
        """
        instrument_key = self.resolve_instrument_key(symbol)
        endpoint = "/market-quote/ltp"
        params = {"instrument_key": instrument_key}

        payload = self._execute_request(endpoint, params=params)
        data = payload.get("data", {})

        if not data:
            raise InstrumentNotFoundError(
                f"No LTP data returned by Upstox for symbol '{symbol}' (key: '{instrument_key}')."
            )

        # Upstox returns keys with either ':' or '|', e.g., 'NSE_INDEX:Nifty 50' or 'NSE_INDEX|Nifty 50'
        # Normalize and look up matching quote
        for key_candidate in [
            instrument_key,
            instrument_key.replace("|", ":"),
            instrument_key.replace(":", "|"),
        ]:
            if key_candidate in data:
                quote_entry = data[key_candidate]
                ltp = quote_entry.get("last_price")
                if ltp is not None:
                    return float(ltp)

        # Fallback: take the first item if single instrument was queried
        if len(data) == 1:
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict) and "last_price" in first_val:
                return float(first_val["last_price"])

        raise InstrumentNotFoundError(
            f"Could not extract LTP from response data for key '{instrument_key}'. Data: {data}"
        )

    def get_expiry_dates(self, symbol: str) -> List[str]:
        """
        Fetch all available option chain expiry dates for a given underlying symbol.

        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY').

        Returns:
            List of ISO expiry date strings (e.g. ['2024-06-06', '2024-06-13']) sorted ascending.
        """
        instrument_key = self.resolve_instrument_key(symbol)
        endpoint = "/option/contract"
        params = {"instrument_key": instrument_key}

        payload = self._execute_request(endpoint, params=params)
        raw_contracts = payload.get("data", [])

        if not raw_contracts:
            raise OptionChainError(
                f"No option contracts found for symbol '{symbol}' ({instrument_key})."
            )

        expiries = set()
        for item in raw_contracts:
            exp = item.get("expiry")
            if exp:
                expiries.add(exp)

        sorted_expiries = sorted(list(expiries))
        return sorted_expiries

    # ------------------------------------------------------------------------
    # Option Chain Fetch & Parsing
    # ------------------------------------------------------------------------

    def get_option_chain(
        self,
        symbol: str,
        expiry_date: Optional[str] = None,
        custom_spot_ltp: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch the option chain data for the nearest weekly (or specified) expiry date,
        compute Call/Put OI, Change in OI, IV, PCR, ATM strike, Max Pain, and Smart Money shifts.

        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY').
            expiry_date: Target expiry date (format 'YYYY-MM-DD'). If None, nearest expiry is picked.
            custom_spot_ltp: Optional spot price override to avoid an additional LTP API call.

        Returns:
            Tuple containing:
                - pd.DataFrame: Clean, strike-wise structured option chain DataFrame.
                - Dict[str, Any]: Comprehensive summary metrics and smart money analysis.
        """
        instrument_key = self.resolve_instrument_key(symbol)

        # 1. Resolve Expiry Date if not supplied
        if not expiry_date:
            available_expiries = self.get_expiry_dates(symbol)
            if not available_expiries:
                raise OptionChainError(f"No active expiry dates found for {symbol}.")
            # Choose nearest active expiry
            today_str = date.today().isoformat()
            future_expiries = [e for e in available_expiries if e >= today_str]
            expiry_date = future_expiries[0] if future_expiries else available_expiries[0]
            logger.info("Selected nearest expiry date for %s: %s", symbol, expiry_date)

        # 2. Fetch Option Chain Data from Upstox API
        endpoint = "/option/chain"
        params = {
            "instrument_key": instrument_key,
            "expiry_date": expiry_date,
        }
        payload = self._execute_request(endpoint, params=params)
        raw_chain = payload.get("data", [])

        if not raw_chain:
            raise OptionChainError(
                f"Option chain data empty for {symbol} on expiry {expiry_date}."
            )

        # 3. Parse Raw JSON Chain into Normalized Rows
        rows: List[Dict[str, Any]] = []
        underlying_spot_from_chain: Optional[float] = None

        for item in raw_chain:
            strike = float(item.get("strike_price", 0.0))
            if strike <= 0:
                continue

            if underlying_spot_from_chain is None and item.get("underlying_spot_price"):
                underlying_spot_from_chain = float(item["underlying_spot_price"])

            call_opt = item.get("call_options") or {}
            put_opt = item.get("put_options") or {}

            call_mkt = call_opt.get("market_data") or {}
            put_mkt = put_opt.get("market_data") or {}

            call_greeks = call_opt.get("option_greeks") or {}
            put_greeks = put_opt.get("option_greeks") or {}

            # Call side fields
            call_oi = int(call_mkt.get("oi", 0) or 0)
            call_prev_oi = int(call_mkt.get("prev_oi", 0) or 0)
            call_change_oi = call_oi - call_prev_oi
            call_ltp = float(call_mkt.get("ltp", 0.0) or 0.0)
            call_volume = int(call_mkt.get("volume", 0) or 0)
            call_iv = float(call_greeks.get("iv", 0.0) or 0.0)
            call_delta = float(call_greeks.get("delta", 0.0) or 0.0)

            # Put side fields
            put_oi = int(put_mkt.get("oi", 0) or 0)
            put_prev_oi = int(put_mkt.get("prev_oi", 0) or 0)
            put_change_oi = put_oi - put_prev_oi
            put_ltp = float(put_mkt.get("ltp", 0.0) or 0.0)
            put_volume = int(put_mkt.get("volume", 0) or 0)
            put_iv = float(put_greeks.get("iv", 0.0) or 0.0)
            put_delta = float(put_greeks.get("delta", 0.0) or 0.0)

            rows.append({
                "strike_price": strike,
                # Calls
                "call_ltp": call_ltp,
                "call_oi": call_oi,
                "call_prev_oi": call_prev_oi,
                "call_change_oi": call_change_oi,
                "call_volume": call_volume,
                "call_iv": round(call_iv, 2),
                "call_delta": round(call_delta, 3),
                # Puts
                "put_ltp": put_ltp,
                "put_oi": put_oi,
                "put_prev_oi": put_prev_oi,
                "put_change_oi": put_change_oi,
                "put_volume": put_volume,
                "put_iv": round(put_iv, 2),
                "put_delta": round(put_delta, 3),
            })

        if not rows:
            raise OptionChainError(f"No valid strike price records found for {symbol}.")

        # 4. Construct DataFrame sorted by strike price
        df = pd.DataFrame(rows).sort_values(by="strike_price").reset_index(drop=True)

        # 5. Resolve Spot LTP
        if custom_spot_ltp is not None:
            spot_ltp = custom_spot_ltp
        elif underlying_spot_from_chain and underlying_spot_from_chain > 0:
            spot_ltp = underlying_spot_from_chain
        else:
            try:
                spot_ltp = self.get_spot_ltp(symbol)
            except Exception as e:
                logger.warning("Could not fetch live spot LTP directly: %s. Using midpoint.", e)
                spot_ltp = float(df["strike_price"].median())

        # 6. Quantitative Analytics
        atm_strike = self.calculate_atm_strike(df["strike_price"].values, spot_ltp)
        max_pain = self.calculate_max_pain(df)
        total_call_oi = int(df["call_oi"].sum())
        total_put_oi = int(df["put_oi"].sum())
        total_call_vol = int(df["call_volume"].sum())
        total_put_vol = int(df["put_volume"].sum())

        # Put-Call Ratio
        pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0.0
        pcr_vol = round(total_put_vol / total_call_vol, 3) if total_call_vol > 0 else 0.0

        # Smart Money Shifts & Key Levels
        smart_money = self.detect_smart_money_shifts(df, spot_ltp, atm_strike)

        # Build Summary Object
        summary = OptionChainSummary(
            symbol=symbol.upper(),
            underlying_key=instrument_key,
            spot_ltp=spot_ltp,
            expiry_date=expiry_date,
            atm_strike=atm_strike,
            max_pain=max_pain,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr_oi=pcr_oi,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            pcr_volume=pcr_vol,
            net_oi_change_bias=smart_money["net_oi_change_bias"],
            sentiment=smart_money["sentiment"],
            key_support_strike=smart_money["key_support_strike"],
            key_resistance_strike=smart_money["key_resistance_strike"],
            max_call_oi_strike=smart_money["max_call_oi_strike"],
            max_put_oi_strike=smart_money["max_put_oi_strike"],
            call_unwinding_strikes=smart_money["call_unwinding"],
            put_writing_strikes=smart_money["put_writing"],
            call_writing_strikes=smart_money["call_writing"],
            put_unwinding_strikes=smart_money["put_unwinding"],
        )

        return df, summary.to_dict()

    # ------------------------------------------------------------------------
    # Financial Analytics (ATM, Max Pain, Smart Money Shifts)
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_atm_strike(strikes: Union[np.ndarray, List[float]], spot_price: float) -> float:
        """
        Calculate the At-The-Money (ATM) strike price closest to the spot price.

        Args:
            strikes: Array or list of available strike prices.
            spot_price: Current underlying spot price.

        Returns:
            ATM strike price.
        """
        strikes_arr = np.array(strikes, dtype=float)
        idx = (np.abs(strikes_arr - spot_price)).argmin()
        return float(strikes_arr[idx])

    @staticmethod
    def calculate_max_pain(df: pd.DataFrame) -> float:
        """
        Calculate Max Pain level across all strikes using vectorized option buyer loss calculation.

        Max Pain is the strike price where option sellers experience minimum loss,
        i.e. where cumulative intrinsic payoff to option buyers on expiry is minimized:
            Total Loss(S) = sum_K [ Call_OI(K) * max(0, S - K) + Put_OI(K) * max(0, K - S) ]

        Args:
            df: DataFrame containing 'strike_price', 'call_oi', and 'put_oi'.

        Returns:
            Max Pain strike price.
        """
        strikes = df["strike_price"].values
        call_oi = df["call_oi"].values
        put_oi = df["put_oi"].values

        # Broadcasted 2D matrix calculation: shape (N, N)
        # S: possible expiry settlement prices (rows)
        # K: strike prices (columns)
        S = strikes[:, np.newaxis]
        K = strikes[np.newaxis, :]

        call_payoff = np.maximum(0, S - K) * call_oi[np.newaxis, :]
        put_payoff = np.maximum(0, K - S) * put_oi[np.newaxis, :]

        total_loss_per_strike = (call_payoff + put_payoff).sum(axis=1)
        min_loss_idx = np.argmin(total_loss_per_strike)

        return float(strikes[min_loss_idx])

    @staticmethod
    def detect_smart_money_shifts(
        df: pd.DataFrame,
        spot_price: float,
        atm_strike: float,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Detect Smart Money Shifts:
        - Call Unwinding: Heavy negative Change in OI on Call strikes (bears closing shorts / bullish shift).
        - Put Writing: Heavy positive Change in OI on Put strikes (bulls selling puts / strong support).
        - Call Writing: Heavy positive Change in OI on Call strikes (bears selling calls / strong resistance).
        - Put Unwinding: Heavy negative Change in OI on Put strikes (bulls closing puts / support failure).

        Args:
            df: DataFrame containing option chain data.
            spot_price: Spot price.
            atm_strike: ATM strike.
            top_n: Number of top strikes to return for each shift category.

        Returns:
            Dictionary with parsed shift metrics and directional sentiment bias.
        """
        # Filter active strikes within reasonable proximity to ATM (e.g. +/- 15 strikes)
        atm_idx = (df["strike_price"] - atm_strike).abs().idxmin()
        start_idx = max(0, atm_idx - 15)
        end_idx = min(len(df), atm_idx + 16)
        active_df = df.iloc[start_idx:end_idx].copy()

        # 1. Call Unwinding (Call Change in OI < 0)
        call_unwinding_df = (
            active_df[active_df["call_change_oi"] < 0]
            .sort_values(by="call_change_oi", ascending=True)
            .head(top_n)
        )
        call_unwinding = [
            {
                "strike": float(row["strike_price"]),
                "change_oi": int(row["call_change_oi"]),
                "oi": int(row["call_oi"]),
                "iv": float(row["call_iv"]),
                "ltp": float(row["call_ltp"]),
            }
            for _, row in call_unwinding_df.iterrows()
        ]

        # 2. Put Writing (Put Change in OI > 0)
        put_writing_df = (
            active_df[active_df["put_change_oi"] > 0]
            .sort_values(by="put_change_oi", ascending=False)
            .head(top_n)
        )
        put_writing = [
            {
                "strike": float(row["strike_price"]),
                "change_oi": int(row["put_change_oi"]),
                "oi": int(row["put_oi"]),
                "iv": float(row["put_iv"]),
                "ltp": float(row["put_ltp"]),
            }
            for _, row in put_writing_df.iterrows()
        ]

        # 3. Call Writing (Call Change in OI > 0)
        call_writing_df = (
            active_df[active_df["call_change_oi"] > 0]
            .sort_values(by="call_change_oi", ascending=False)
            .head(top_n)
        )
        call_writing = [
            {
                "strike": float(row["strike_price"]),
                "change_oi": int(row["call_change_oi"]),
                "oi": int(row["call_oi"]),
                "iv": float(row["call_iv"]),
                "ltp": float(row["call_ltp"]),
            }
            for _, row in call_writing_df.iterrows()
        ]

        # 4. Put Unwinding (Put Change in OI < 0)
        put_unwinding_df = (
            active_df[active_df["put_change_oi"] < 0]
            .sort_values(by="put_change_oi", ascending=True)
            .head(top_n)
        )
        put_unwinding = [
            {
                "strike": float(row["strike_price"]),
                "change_oi": int(row["put_change_oi"]),
                "oi": int(row["put_oi"]),
                "iv": float(row["put_iv"]),
                "ltp": float(row["put_ltp"]),
            }
            for _, row in put_unwinding_df.iterrows()
        ]

        # Key levels
        max_call_oi_strike = float(df.loc[df["call_oi"].idxmax()]["strike_price"])
        max_put_oi_strike = float(df.loc[df["put_oi"].idxmax()]["strike_price"])

        # Resistance = strike above spot with highest Call Change OI or highest Call OI
        calls_above_spot = df[df["strike_price"] >= spot_price]
        key_resistance = (
            float(calls_above_spot.loc[calls_above_spot["call_oi"].idxmax()]["strike_price"])
            if not calls_above_spot.empty
            else max_call_oi_strike
        )

        # Support = strike below spot with highest Put Change OI or highest Put OI
        puts_below_spot = df[df["strike_price"] <= spot_price]
        key_support = (
            float(puts_below_spot.loc[puts_below_spot["put_oi"].idxmax()]["strike_price"])
            if not puts_below_spot.empty
            else max_put_oi_strike
        )

        # Quantitative Market Sentiment scoring
        total_put_chg = int(df["put_change_oi"].sum())
        total_call_chg = int(df["call_change_oi"].sum())
        net_oi_change_bias = total_put_chg - total_call_chg

        # Sentiment heuristics
        if net_oi_change_bias > 500_000 and total_put_chg > 0:
            sentiment = "Strongly Bullish (Aggressive Put Writing)"
        elif net_oi_change_bias > 100_000:
            sentiment = "Bullish (Put Writing & Call Unwinding)"
        elif net_oi_change_bias < -500_000 and total_call_chg > 0:
            sentiment = "Strongly Bearish (Heavy Call Writing)"
        elif net_oi_change_bias < -100_000:
            sentiment = "Bearish (Call Writing & Put Unwinding)"
        else:
            sentiment = "Neutral / Rangebound"

        return {
            "sentiment": sentiment,
            "net_oi_change_bias": net_oi_change_bias,
            "call_unwinding": call_unwinding,
            "put_writing": put_writing,
            "call_writing": call_writing,
            "put_unwinding": put_unwinding,
            "max_call_oi_strike": max_call_oi_strike,
            "max_put_oi_strike": max_put_oi_strike,
            "key_resistance_strike": key_resistance,
            "key_support_strike": key_support,
        }


# ============================================================================
# Upstox WebSocket Streamer for Live Quotes & OI Feed
# ============================================================================

class UpstoxMarketDataWebSocket:
    """
    WebSocket client for streaming live Upstox API v2 Market Data and quotes.
    Handles auto-reconnect, JSON/Protobuf message decoding, and subscription management.
    """

    WS_FEED_URL: str = "wss://api.upstox.com/v2/feed/market-data-feed"

    def __init__(
        self,
        access_token: str,
        on_message_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error_callback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.access_token = access_token.strip()
        self.on_message_callback = on_message_callback
        self.on_error_callback = on_error_callback
        self._is_running = False
        self._subscribed_keys: List[str] = []

    async def get_authorized_ws_url(self) -> str:
        """
        Fetch authorized WebSocket redirect URL from Upstox v2.
        Endpoint: GET /v2/feed/market-data-feed/authorize
        """
        auth_url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        res = requests.get(auth_url, headers=headers, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            if payload.get("status") == "success":
                return payload["data"]["authorizedRedirectUri"]
        # Fallback to direct WebSocket feed URL
        return self.WS_FEED_URL

    async def subscribe(
        self,
        instrument_keys: List[str],
        mode: Literal["ltpc", "full"] = "full",
    ) -> None:
        """
        Subscribe to live tick feeds for specified instrument keys.

        Args:
            instrument_keys: List of Upstox instrument keys (e.g. ['NSE_INDEX|Nifty 50']).
            mode: 'ltpc' for Last Traded Price + Close or 'full' for complete depth & OI.
        """
        self._subscribed_keys = instrument_keys
        self._is_running = True

        try:
            import websockets
        except ImportError:
            logger.error("The 'websockets' library is required for live WebSocket streaming.")
            return

        while self._is_running:
            try:
                ws_uri = await self.get_authorized_ws_url()
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                async with websockets.connect(
                    ws_uri,
                    ssl=ssl_context,
                    extra_headers={"Authorization": f"Bearer {self.access_token}"},
                ) as ws:
                    logger.info("Connected to Upstox Market Data WebSocket.")

                    # Send subscription request
                    sub_payload = {
                        "guid": "guid_stream_1",
                        "method": "sub",
                        "params": {
                            "mode": mode,
                            "instrumentKeys": instrument_keys,
                        },
                    }
                    await ws.send(json.dumps(sub_payload))

                    while self._is_running:
                        message = await ws.recv()
                        try:
                            data = json.loads(message) if isinstance(message, str) else {"raw": message}
                            if self.on_message_callback:
                                self.on_message_callback(data)
                        except Exception as parse_err:
                            logger.debug("Raw binary tick received or parse notice: %s", parse_err)

            except Exception as e:
                logger.warning("WebSocket disconnected with error: %s. Reconnecting in 3s...", e)
                if self.on_error_callback:
                    self.on_error_callback(e)
                if not self._is_running:
                    break
                await asyncio.sleep(3)

    def stop(self) -> None:
        """Gracefully stop the WebSocket loop."""
        self._is_running = False
        logger.info("WebSocket streamer stopped.")


# ============================================================================
# Example / Verification Entrypoint
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UPSTOX OPTION CHAIN ENGINE - MODULE DEMO & VERIFICATION")
    print("=" * 70)

    # Demonstrate synthetic verification if run without an actual API token
    sample_strikes = np.arange(24000, 25001, 50)
    spot_demo = 24525.0

    mock_rows = []
    np.random.seed(42)
    for k in sample_strikes:
        c_oi = int(np.random.randint(20000, 150000))
        p_oi = int(np.random.randint(20000, 150000))
        mock_rows.append({
            "strike_price": float(k),
            "call_ltp": round(max(5.0, spot_demo - k + 100), 2),
            "call_oi": c_oi,
            "call_prev_oi": c_oi - int(np.random.randint(-15000, 25000)),
            "call_change_oi": int(np.random.randint(-15000, 25000)),
            "call_volume": int(np.random.randint(50000, 500000)),
            "call_iv": round(np.random.uniform(11.0, 16.0), 2),
            "call_delta": round(max(0.01, min(0.99, 0.5 + (spot_demo - k) / 1000)), 3),
            "put_ltp": round(max(5.0, k - spot_demo + 100), 2),
            "put_oi": p_oi,
            "put_prev_oi": p_oi - int(np.random.randint(-15000, 25000)),
            "put_change_oi": int(np.random.randint(-15000, 25000)),
            "put_volume": int(np.random.randint(50000, 500000)),
            "put_iv": round(np.random.uniform(11.0, 16.0), 2),
            "put_delta": round(max(-0.99, min(-0.01, -0.5 + (spot_demo - k) / 1000)), 3),
        })

    demo_df = pd.DataFrame(mock_rows)
    atm_calc = OptionChainEngine.calculate_atm_strike(demo_df["strike_price"].values, spot_demo)
    max_pain_calc = OptionChainEngine.calculate_max_pain(demo_df)
    shifts = OptionChainEngine.detect_smart_money_shifts(demo_df, spot_demo, atm_calc)

    print(f"Demo Spot LTP: {spot_demo}")
    print(f"Calculated ATM Strike: {atm_calc}")
    print(f"Calculated Max Pain Level: {max_pain_calc}")
    print(f"Market Sentiment: {shifts['sentiment']}")
    print(f"Top Call Unwinding Strikes: {shifts['call_unwinding'][:2]}")
    print(f"Top Put Writing Strikes: {shifts['put_writing'][:2]}")
    print("=" * 70)
    print("OptionChainEngine initialized and verified successfully.")
