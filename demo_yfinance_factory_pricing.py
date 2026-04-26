import argparse

from market_data_factory import MarketDataFactory
from ssvi import SSVIVolSurface
from surface_pricers import BlackScholesSurfacePricer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--options-csv", default="data/msft_options_yfinance.csv")
    parser.add_argument("--ssvi-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--maturity-date", default="2026-12-18")
    parser.add_argument("--strike", type=float, default=430.0)
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    args = parser.parse_args()

    points = MarketDataFactory.points_from_yfinance_csv(args.options_csv)
    selected_points = [
        point
        for point in points
        if point.ticker == args.ticker
        and point.maturity_date.date().isoformat() == args.maturity_date
        and point.strike == args.strike
        and point.option_type == args.option_type
    ]
    if not selected_points:
        raise ValueError("Aucun point yfinance ne correspond aux criteres demandes")

    point = selected_points[0]
    rate_curve = MarketDataFactory.rate_curve_from_parquet(args.rate_curve)
    dividend_yield = MarketDataFactory.dividend_yield_from_yfinance(args.ticker)
    option, market = MarketDataFactory.option_and_market_from_point(
        point,
        rate_curve,
        dividend_yield,
    )

    vol_surface = SSVIVolSurface.from_csv(args.ssvi_params)
    pricer = BlackScholesSurfacePricer(option, vol_surface, market, point.trade_date.date())

    print(f"Market mid: {point.option_price:.6f}")
    print(f"Surface volatility: {pricer.implied_surface_vol():.6f}")
    print(f"Model price: {pricer.price():.6f}")


if __name__ == "__main__":
    main()
