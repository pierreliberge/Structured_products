import argparse
from datetime import datetime

from markets import Market
from products import ExerciseType, Option, OptionType
from rate_curve_loader import RateCurveParquetLoader
from ssvi import SSVIVolSurface
from surface_pricers import BlackScholesSurfacePricer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=float, default=424.6199951171875)
    parser.add_argument("--strike", type=float, default=430.0)
    parser.add_argument("--maturity-date", default="2026-12-18")
    parser.add_argument("--pricing-date", default="2026-04-25")
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    parser.add_argument("--dividend-yield", type=float, default=0.0086)
    parser.add_argument("--ssvi-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    args = parser.parse_args()

    pricing_date = datetime.strptime(args.pricing_date, "%Y-%m-%d").date()
    maturity_date = datetime.strptime(args.maturity_date, "%Y-%m-%d").date()

    rate_curve = RateCurveParquetLoader(args.rate_curve).load_latest_curve()
    market = Market(args.spot, rate_curve, q=args.dividend_yield)
    vol_surface = SSVIVolSurface.from_csv(args.ssvi_params)

    option = Option(
        maturity_date,
        args.strike,
        OptionType(args.option_type),
        ExerciseType.EUROPEAN,
        "ACT/365",
    )
    pricer = BlackScholesSurfacePricer(option, vol_surface, market, pricing_date)

    print(f"Surface volatility: {pricer.implied_surface_vol():.6f}")
    print(f"Price: {pricer.price():.6f}")


if __name__ == "__main__":
    main()
