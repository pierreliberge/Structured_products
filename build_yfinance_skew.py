import argparse

from implied_vol import (
    ImpliedVolCalculator,
    save_implied_vols,
)
from plot_utils import plot_skew
from rate_curve_loader import RateCurveParquetLoader
from vol_surface import keep_maturity, keep_otm_options
from yfinance_options_loader import YFinanceOptionsLoader, save_calibration_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--rate-country", default="United States")
    parser.add_argument("--dividend-yield", type=float, default=None)
    parser.add_argument("--maturity-date", default=None)
    parser.add_argument("--min-maturity-days", type=int, default=30)
    parser.add_argument("--max-maturity-days", type=int, default=365)
    parser.add_argument("--max-expirations", type=int, default=None)
    parser.add_argument("--min-price", type=float, default=0.01)
    parser.add_argument("--output-options", default="data/msft_options_yfinance.csv")
    parser.add_argument("--output-iv", default="data/implied_vol_skew_2.csv")
    parser.add_argument("--output-plot", default=None)
    args = parser.parse_args()

    dividend_yield = args.dividend_yield
    if dividend_yield is None:
        dividend_yield = YFinanceOptionsLoader.get_dividend_yield(args.ticker)
        print(f"Dividend yield from yfinance: {dividend_yield:.6f}")

    loader = YFinanceOptionsLoader(args.ticker)
    points = loader.load(
        min_maturity_days=args.min_maturity_days,
        max_maturity_days=args.max_maturity_days,
        max_expirations=args.max_expirations,
        min_price=args.min_price,
    )
    save_calibration_points(points, args.output_options)

    skew_points = keep_maturity(points, args.maturity_date)
    rate = args.rate
    if rate is None:
        rate_curve = RateCurveParquetLoader(args.rate_curve).load_latest_curve(
            country=args.rate_country,
            compounding="continuous",
        )
        rate = rate_curve.get_rate(skew_points[0].maturity)
        print(f"Rate from parquet curve: {rate:.6f}")

    skew_points = keep_otm_options(skew_points, rate, dividend_yield)
    implied_vol_points = ImpliedVolCalculator.compute_points(
        skew_points,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    implied_vol_points = sorted(implied_vol_points, key=lambda point: point.strike)

    save_implied_vols(implied_vol_points, args.output_iv)
    if args.output_plot is not None:
        plot_skew(implied_vol_points, args.output_plot, rate, dividend_yield)

    print(f"{len(points)} yfinance option points saved: {args.output_options}")
    print(f"{len(implied_vol_points)} implied volatility points saved: {args.output_iv}")
    if args.output_plot is not None:
        print(f"Plot: {args.output_plot}")


if __name__ == "__main__":
    main()
