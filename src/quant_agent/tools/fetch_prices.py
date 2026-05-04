from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quant_agent.data.cache import get_prices
from quant_agent.tools.base import Tool


class FetchPricesInput(BaseModel):
    ticker: str = Field(default="SPY", description="Ticker symbol. Default 'SPY'.")
    start: str = Field(description="Start date YYYY-MM-DD (inclusive).")
    end: str = Field(description="End date YYYY-MM-DD (exclusive).")


class FetchPricesTool(Tool):
    name = "fetch_prices"
    description = (
        "Fetch daily OHLCV price data for a ticker between two dates. "
        "Returns a summary (date range, number of rows, first/last close). "
        "The actual data is cached internally and consumed by run_backtest."
    )
    input_model = FetchPricesInput

    def run(self, params: FetchPricesInput) -> dict[str, Any]:
        df = get_prices(params.ticker, params.start, params.end)
        return {
            "ticker": params.ticker,
            "actual_start": str(df.index.min().date()),
            "actual_end": str(df.index.max().date()),
            "n_rows": len(df),
            "first_close": round(float(df["Close"].iloc[0]), 4),
            "last_close": round(float(df["Close"].iloc[-1]), 4),
        }
