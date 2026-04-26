import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssvi import SSVIVolSurface


def build_grid(
    surface: SSVIVolSurface,
    market_surface_path: str,
    n_maturities: int,
    n_strikes: int,
) -> pd.DataFrame:
    market_df = pd.read_csv(market_surface_path, sep=";")

    min_maturity = market_df["maturity"].min()
    max_maturity = market_df["maturity"].max()
    min_strike = market_df["strike"].min()
    max_strike = market_df["strike"].max()
    spot = float(market_df["spot"].iloc[0])
    rate = float(market_df["rate"].median())
    dividend_yield = float(market_df["dividend_yield"].median())

    maturities = np.linspace(min_maturity, max_maturity, n_maturities)
    strikes = np.linspace(min_strike, max_strike, n_strikes)

    rows = []
    for maturity in maturities:
        forward = spot * np.exp((rate - dividend_yield) * maturity)
        for strike in strikes:
            rows.append(
                {
                    "maturity": maturity,
                    "strike": strike,
                    "forward": forward,
                    "ssvi_implied_vol": surface.get_vol(maturity, strike, forward),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--market-surface", default="data/implied_vol_surface.csv")
    parser.add_argument("--output", default="data/ssvi_surface_grid.csv")
    parser.add_argument("--n-maturities", type=int, default=20)
    parser.add_argument("--n-strikes", type=int, default=80)
    args = parser.parse_args()

    surface = SSVIVolSurface.from_csv(args.params)
    grid_df = build_grid(
        surface,
        args.market_surface,
        args.n_maturities,
        args.n_strikes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_path, sep=";", index=False)
    print(f"SSVI surface grid saved: {args.output}")


if __name__ == "__main__":
    main()
