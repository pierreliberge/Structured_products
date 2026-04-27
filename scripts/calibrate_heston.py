import argparse
from pathlib import Path

from calibration.heston import (
    HestonFourierCalibrator,
    HestonIVHeuristicCalibrator,
    load_implied_vol_points,
)
from calibration.implied_vol import ImpliedVolCalculator
from market.options_loader import BloombergOptionsLoader
from market.rate_curve_loader import RateCurveParquetLoader
from market.vol_surface import build_surface_points, keep_latest_trade_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/total_iv_surface_bloomberg.csv")
    parser.add_argument("--input-kind", choices=["options", "implied_vol"], default="implied_vol")
    parser.add_argument("--ticker", default="TTE")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--rate-country", default="France")
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--min-maturity-days", type=int, default=7)
    parser.add_argument("--max-calibration-points", type=int, default=80)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--integration-upper-bound", type=float, default=80.0)
    parser.add_argument("--integration-limit", type=int, default=80)
    parser.add_argument("--method", choices=["fourier", "heuristic"], default="fourier")
    parser.add_argument("--output", default="data/output/total_heston_params.csv")
    args = parser.parse_args()

    if args.input_kind == "implied_vol":
        points = load_implied_vol_points(args.input)
        if args.ticker is not None:
            points = [point for point in points if point.ticker == args.ticker]
        rate_curve = RateCurveParquetLoader(args.rate_curve).load_latest_curve(
            args.rate_country
        )
        for point in points:
            if point.rate is None:
                point.rate = rate_curve.get_rate(point.maturity)
            if point.dividend_yield is None:
                point.dividend_yield = args.dividend_yield
    else:
        raw_points = BloombergOptionsLoader(args.input).load(
            ticker=args.ticker,
            min_price=0.0,
            min_maturity=args.min_maturity_days / 365,
        )
        raw_points = keep_latest_trade_date(raw_points)
        rate_curve = RateCurveParquetLoader(args.rate_curve).load_latest_curve(
            args.rate_country
        )
        points = build_surface_points(
            raw_points,
            rate_curve,
            args.dividend_yield,
            ImpliedVolCalculator(),
        )

    if args.method == "heuristic":
        params = HestonIVHeuristicCalibrator(points).estimate()
        calibration_points = len(points)
    else:
        calibrator = HestonFourierCalibrator(
            points,
            max_points=args.max_calibration_points,
            integration_upper_bound=args.integration_upper_bound,
            integration_limit=args.integration_limit,
        )
        params = calibrator.calibrate(max_iterations=args.max_iterations)
        calibration_points = len(calibrator.points)

    output_path = Path(args.output)
    params.save_csv(output_path)

    print(f"{len(points)} points IV utilises.")
    print(f"{calibration_points} points utilises pour la calibration.")
    print(f"Parametres Heston sauvegardes: {output_path}")
    print(params.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
