import argparse
from datetime import datetime

from dupire import DupireLocalVolSurface
from markets import Market
from market_data_factory import MarketDataFactory
from models import LocalVolModel
from pricers import McPricer
from products import ExerciseType, Option, OptionType
from ssvi import SSVIVolSurface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=float, default=424.6199951171875)
    parser.add_argument("--strike", type=float, default=430.0)
    parser.add_argument("--maturity-date", default="2026-12-18")
    parser.add_argument("--pricing-date", default="2026-04-25")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--ssvi-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--n-paths", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=100)
    args = parser.parse_args()

    pricing_date = datetime.strptime(args.pricing_date, "%Y-%m-%d").date()
    maturity_date = datetime.strptime(args.maturity_date, "%Y-%m-%d").date()

    rate_curve = MarketDataFactory.rate_curve_from_parquet(args.rate_curve)
    dividend_yield = MarketDataFactory.dividend_yield_from_yfinance(args.ticker)
    market = Market(args.spot, rate_curve, q=dividend_yield)

    ssvi_surface = SSVIVolSurface.from_csv(args.ssvi_params)
    dupire_surface = DupireLocalVolSurface(
        ssvi_surface,
        args.spot,
        rate_curve,
        dividend_yield,
    )
    model = LocalVolModel(dupire_surface)

    option = Option(
        maturity_date,
        args.strike,
        OptionType(args.option_type),
        ExerciseType.EUROPEAN,
        "ACT/365",
    )
    pricer = McPricer(
        option,
        model,
        market,
        pricing_date,
        args.n_paths,
        args.n_steps,
    )

    print(f"Local vol MC price: {pricer.price():.6f}")


if __name__ == "__main__":
    main()
