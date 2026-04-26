import argparse

from implied_vol import ImpliedVolCalculator, save_implied_vols
from plot_utils import plot_surface_skews
from rate_curve_loader import RateCurveParquetLoader
from vol_surface import build_surface_points, group_by_maturity
from yfinance_options_loader import YFinanceOptionsLoader, save_calibration_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--rate-country", default="United States")
    parser.add_argument("--dividend-yield", type=float, default=None)
    parser.add_argument("--min-maturity-days", type=int, default=30)
    parser.add_argument("--max-maturity-days", type=int, default=365)
    parser.add_argument("--max-expirations", type=int, default=None)
    parser.add_argument("--min-price", type=float, default=0.01)
    parser.add_argument("--output-options", default="data/msft_options_yfinance.csv")
    parser.add_argument("--output-surface", default="data/implied_vol_surface.csv")
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--max-plot-maturities", type=int, default=8)
    args = parser.parse_args()

    dividend_yield = args.dividend_yield
    if dividend_yield is None:
        dividend_yield = YFinanceOptionsLoader.get_dividend_yield(args.ticker)
        print(f"Dividend yield from yfinance: {dividend_yield:.6f}")

    options_loader = YFinanceOptionsLoader(args.ticker)
    points = options_loader.load(
        min_maturity_days=args.min_maturity_days,
        max_maturity_days=args.max_maturity_days,
        max_expirations=args.max_expirations,
        min_price=args.min_price,
    )
    save_calibration_points(points, args.output_options)

    rate_curve = RateCurveParquetLoader(args.rate_curve).load_latest_curve(
        country=args.rate_country,
        compounding="continuous",
    )
    surface_points = build_surface_points(
        points,
        rate_curve,
        dividend_yield,
        ImpliedVolCalculator,
    )

    save_implied_vols(surface_points, args.output_surface)
    if args.output_plot is not None:
        plot_surface_skews(surface_points, args.output_plot, args.max_plot_maturities)

    maturities_count = len(group_by_maturity(surface_points))
    print(f"{len(points)} yfinance option points saved: {args.output_options}")
    print(f"{len(surface_points)} implied volatility surface points saved: {args.output_surface}")
    print(f"{maturities_count} maturities in surface.")
    if args.output_plot is not None:
        print(f"Plot: {args.output_plot}")


if __name__ == "__main__":
    main()
