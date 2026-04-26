import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dupire import DupireLocalVolSurface
from market_data_factory import MarketDataFactory
from ssvi import SSVIVolSurface


def build_local_vol_grid(
    dupire_surface: DupireLocalVolSurface,
    market_df: pd.DataFrame,
    n_maturities: int,
    n_strikes: int,
) -> pd.DataFrame:
    min_maturity = market_df["maturity"].min()
    max_maturity = market_df["maturity"].max()
    min_strike = market_df["strike"].quantile(0.05)
    max_strike = market_df["strike"].quantile(0.95)

    maturities = np.linspace(min_maturity, max_maturity, n_maturities)
    strikes = np.linspace(min_strike, max_strike, n_strikes)

    rows = []
    for maturity in maturities:
        for strike in strikes:
            try:
                local_vol = dupire_surface.get_local_vol(maturity, strike)
                error = ""
            except ValueError as exc:
                local_vol = np.nan
                error = str(exc)

            rows.append(
                {
                    "maturity": maturity,
                    "strike": strike,
                    "local_vol": local_vol,
                    "error": error,
                }
            )

    return pd.DataFrame(rows)


def print_diagnostics(grid_df: pd.DataFrame) -> None:
    valid = grid_df.dropna(subset=["local_vol"])
    invalid = grid_df[grid_df["local_vol"].isna()]

    print(f"Grid points: {len(grid_df)}")
    print(f"Valid local vols: {len(valid)}")
    print(f"Invalid local vols: {len(invalid)}")

    if not valid.empty:
        print(f"Min local vol: {valid['local_vol'].min():.6f}")
        print(f"Median local vol: {valid['local_vol'].median():.6f}")
        print(f"Max local vol: {valid['local_vol'].max():.6f}")
        print(f"Local vol > 100%: {(valid['local_vol'] > 1.0).sum()}")
        print(f"Local vol > 150%: {(valid['local_vol'] > 1.5).sum()}")

    if not invalid.empty:
        print("Most common errors:")
        print(invalid["error"].value_counts().head().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssvi-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--market-surface", default="data/implied_vol_surface.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--output", default="data/dupire_local_vol_grid.csv")
    parser.add_argument("--n-maturities", type=int, default=15)
    parser.add_argument("--n-strikes", type=int, default=50)
    parser.add_argument("--dt-days", type=float, default=1.0)
    parser.add_argument("--relative-dk", type=float, default=0.005)
    args = parser.parse_args()

    market_df = pd.read_csv(args.market_surface, sep=";")
    spot = float(market_df["spot"].iloc[0])
    dividend_yield = MarketDataFactory.dividend_yield_from_yfinance(args.ticker)
    rate_curve = MarketDataFactory.rate_curve_from_parquet(args.rate_curve)
    ssvi_surface = SSVIVolSurface.from_csv(args.ssvi_params)

    dupire_surface = DupireLocalVolSurface(
        ssvi_surface,
        spot,
        rate_curve,
        dividend_yield,
        dt=args.dt_days / 365,
        relative_dk=args.relative_dk,
    )

    grid_df = build_local_vol_grid(
        dupire_surface,
        market_df,
        args.n_maturities,
        args.n_strikes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_path, sep=";", index=False)

    print_diagnostics(grid_df)
    print(f"Local vol grid saved: {args.output}")


if __name__ == "__main__":
    main()
