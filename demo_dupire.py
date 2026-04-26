import argparse

import pandas as pd

from dupire import DupireLocalVolSurface
from market_data_factory import MarketDataFactory
from ssvi import SSVIVolSurface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssvi-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--market-surface", default="data/implied_vol_surface.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--maturity", type=float, default=0.65)
    parser.add_argument("--strike", type=float, default=430.0)
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
    )

    local_vol = dupire_surface.get_local_vol(args.maturity, args.strike)
    print(f"Dupire local volatility: {local_vol:.6f}")


if __name__ == "__main__":
    main()
