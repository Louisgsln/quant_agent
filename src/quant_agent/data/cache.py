"""yfinance wrapper with parquet disk cache.

Why a cache:
- Determinism for evals: a backtest run today must give the same numbers tomorrow.
- Offline development: dev without internet, no rate-limiting concerns.
- Speed: parquet round-trip is ~50x faster than yfinance HTTP fetch.

Cache key is a hash of (ticker, start, end, interval). Any param change = cache miss.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from quant_agent.config import settings
from quant_agent.logging_setup import get_logger

log = get_logger(__name__)


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    """Stable cache file path for a given (ticker, range, interval) tuple."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    raw = f"{ticker}|{start}|{end}|{interval}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return settings.cache_dir / f"{ticker}_{digest}.parquet"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    reraise=True,
)
def _download(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Hit yfinance with retries and exponential backoff.

    yfinance returns MultiIndex columns when given multiple tickers. We always
    pass a single ticker but flatten defensively in case yfinance changes behavior.
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} in [{start}, {end}]")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV data, with parquet disk cache.

    Args:
        ticker:    e.g. "SPY", "AAPL", "^GSPC".
        start:     ISO date "YYYY-MM-DD" (inclusive).
        end:       ISO date "YYYY-MM-DD" (exclusive, per yfinance convention).
        interval:  "1d" by default. "1h", "1wk", etc. supported.
        use_cache: Set to False to force a fresh download.

    Returns:
        DataFrame with DatetimeIndex (tz-naive) and columns
        ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    path = _cache_path(ticker, start, end, interval)

    if use_cache and path.exists():
        log.debug("cache_hit", ticker=ticker, path=str(path))
        return pd.read_parquet(path)

    log.info("cache_miss_downloading", ticker=ticker, start=start, end=end, interval=interval)
    df = _download(ticker, start, end, interval)
    df.to_parquet(path)
    log.info("cache_written", ticker=ticker, rows=len(df), path=str(path))
    return df
