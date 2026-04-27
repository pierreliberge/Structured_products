from math import erf, exp, log, sqrt
from pathlib import Path
import argparse
import csv

from market.market_data import ImpliedVolPoint, OptionCalibrationPoint
from market.options_loader import BloombergOptionsLoader
from visualization.plot_utils import plot_skew
from market.vol_surface import keep_maturity


class ImpliedVolCalculator:
    @staticmethod
    def black_scholes_price(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        option_type: str,
        dividend_yield: float = 0.0,
    ) -> float:
        if maturity <= 0:
            if option_type == "call":
                return max(spot - strike, 0.0)
            if option_type == "put":
                return max(strike - spot, 0.0)
            raise ValueError("option_type must be 'call' or 'put'")

        if volatility <= 0:
            raise ValueError("volatility must be positive")

        d1 = (
            log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility**2) * maturity
        ) / (volatility * sqrt(maturity))
        d2 = d1 - volatility * sqrt(maturity)

        df_r = exp(-rate * maturity)
        df_q = exp(-dividend_yield * maturity)

        if option_type == "call":
            return spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2)
        if option_type == "put":
            return strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'")

    @classmethod
    def implied_volatility(
        cls,
        point: OptionCalibrationPoint,
        rate: float,
        dividend_yield: float = 0.0,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float | None:
        low = 1e-6
        high = 5.0

        price_low = cls.black_scholes_price(
            point.spot,
            point.strike,
            point.maturity,
            rate,
            low,
            point.option_type,
            dividend_yield,
        )
        price_high = cls.black_scholes_price(
            point.spot,
            point.strike,
            point.maturity,
            rate,
            high,
            point.option_type,
            dividend_yield,
        )

        if point.option_price < price_low - tolerance:
            return None
        if point.option_price > price_high + tolerance:
            return None

        for _ in range(max_iterations):
            mid = (low + high) / 2
            price_mid = cls.black_scholes_price(
                point.spot,
                point.strike,
                point.maturity,
                rate,
                mid,
                point.option_type,
                dividend_yield,
            )

            if abs(price_mid - point.option_price) < tolerance:
                return mid

            if price_mid > point.option_price:
                high = mid
            else:
                low = mid

        return (low + high) / 2

    @classmethod
    def compute_points(
        cls,
        points: list[OptionCalibrationPoint],
        rate: float,
        dividend_yield: float = 0.0,
    ) -> list[ImpliedVolPoint]:
        implied_vol_points = []
        for point in points:
            implied_vol = cls.implied_volatility(point, rate, dividend_yield)
            if implied_vol is None:
                continue
            implied_vol_points.append(
                ImpliedVolPoint(
                    trade_date=point.trade_date,
                    maturity_date=point.maturity_date,
                    maturity=point.maturity,
                    strike=point.strike,
                    spot=point.spot,
                    option_price=point.option_price,
                    option_type=point.option_type,
                    implied_vol=implied_vol,
                    ticker=point.ticker,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    forward=point.forward
                    if point.forward is not None
                    else point.spot * exp((rate - dividend_yield) * point.maturity),
                )
            )
        return implied_vol_points


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def save_implied_vols(points: list[ImpliedVolPoint], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(
            [
                "trade_date",
                "maturity_date",
                "maturity",
                "strike",
                "spot",
                "option_price",
                "option_type",
                "implied_vol",
                "ticker",
                "rate",
                "dividend_yield",
                "forward",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.trade_date.date(),
                    point.maturity_date.date(),
                    point.maturity,
                    point.strike,
                    point.spot,
                    point.option_price,
                    point.option_type,
                    point.implied_vol,
                    point.ticker,
                    point.rate,
                    point.dividend_yield,
                    point.forward,
                ]
            )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TTE")
    parser.add_argument("--rate", type=float, default=0.0244)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--maturity-date", default=None)
    parser.add_argument("--input", default="data/Total_vol_final.xlsx")
    parser.add_argument("--source", choices=["bloomberg"], default="bloomberg")
    parser.add_argument("--min-maturity-days", type=int, default=30)
    parser.add_argument("--max-maturity-days", type=int, default=None)
    parser.add_argument("--max-expirations", type=int, default=None)
    parser.add_argument("--output-csv", default="data/total_iv_surface_bloomberg.csv")
    parser.add_argument("--output-plot", default=None)
    args = parser.parse_args()

    loader = BloombergOptionsLoader(args.input)
    raw_points = loader.load_implied_vol_points(
        ticker=args.ticker,
        min_maturity=args.min_maturity_days / 365,
        only_otm=True,
    )
    implied_vol_points = (
        keep_maturity(raw_points, args.maturity_date)
        if args.maturity_date is not None
        else raw_points
    )
    implied_vol_points = sorted(implied_vol_points, key=lambda point: point.strike)

    save_implied_vols(implied_vol_points, args.output_csv)
    if args.output_plot is not None:
        plot_skew(implied_vol_points, args.output_plot, args.rate, args.dividend_yield)

    print(f"{len(implied_vol_points)} implied volatility points saved.")
    print(f"CSV: {args.output_csv}")
    if args.output_plot is not None:
        print(f"Plot: {args.output_plot}")


if __name__ == "__main__":
    main()



