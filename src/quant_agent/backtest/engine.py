from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest output series.

    All series are aligned on the input price index. Equity curves start at 1.0.
    """

    returns: pd.Series  # daily strategy returns, net of costs
    positions: pd.Series  # daily applied positions (after shift)
    equity: pd.Series  # cumulative equity, starts at 1.0
    benchmark_returns: pd.Series  # buy-and-hold daily returns
    benchmark_equity: pd.Series  # buy-and-hold equity curve

    def summary(self) -> dict[str, float]:
        """Quick high-level summary, useful for logging."""
        return {
            "n_days": len(self.returns),
            "final_equity": float(self.equity.iloc[-1]),
            "benchmark_final_equity": float(self.benchmark_equity.iloc[-1]),
            "turnover_total": float(self.positions.diff().abs().sum()),
        }


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = 1.0,
) -> BacktestResult:
    """Apply a position series to prices, with simple linear transaction costs.

    Args:
        prices:   DataFrame with a 'Close' column, DatetimeIndex.
        positions: Series in {0, 1} aligned on prices.index. Decided at close
                   of day t.
        cost_bps: Cost per unit of turnover, in basis points (1 bps = 0.01%).
                  Applied to abs(position change). Default 1 bps is realistic
                  for liquid ETFs like SPY.

    Returns:
        BacktestResult with strategy and benchmark series.
    """
    if "Close" not in prices.columns:
        raise ValueError("prices must contain a 'Close' column")
    if not prices.index.equals(positions.index):
        raise ValueError("prices and positions must share the same index")

    close = prices["Close"]
    asset_rets = close.pct_change().fillna(0)

    # Look-ahead protection: position decided at close of t applies to return t+1.
    applied_pos = positions.shift(1).fillna(0)

    # Transaction costs proportional to position change.
    turnover = applied_pos.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10_000)

    strat_rets = applied_pos * asset_rets - costs
    equity = (1 + strat_rets).cumprod()

    bench_equity = (1 + asset_rets).cumprod()

    return BacktestResult(
        returns=strat_rets,
        positions=applied_pos,
        equity=equity,
        benchmark_returns=asset_rets,
        benchmark_equity=bench_equity,
    )
