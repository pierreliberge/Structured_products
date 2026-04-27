from math import exp
from pathlib import Path

import matplotlib.pyplot as plt

from market.market_data import ImpliedVolPoint
from market.vol_surface import group_by_maturity


def plot_skew(
    points: list[ImpliedVolPoint],
    output_path: str | Path,
    rate: float | None = None,
    dividend_yield: float = 0.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = sorted(points, key=lambda point: point.strike)
    put_points = [point for point in points if point.option_type == "put"]
    call_points = [point for point in points if point.option_type == "call"]

    plt.figure(figsize=(9, 5))
    if put_points:
        plt.plot(
            [point.strike for point in put_points],
            [point.implied_vol for point in put_points],
            marker="o",
            linestyle="-",
            label="OTM puts",
        )
    if call_points:
        plt.plot(
            [point.strike for point in call_points],
            [point.implied_vol for point in call_points],
            marker="o",
            linestyle="-",
            label="OTM calls",
        )

    first = points[0]
    plt.axvline(first.spot, color="gray", linestyle="--", label="Spot")
    if first.forward is not None:
        forward = first.forward
        plt.axvline(forward, color="black", linestyle=":", label="ATM forward")
    elif rate is not None:
        forward = first.spot * exp((rate - dividend_yield) * first.maturity)
        plt.axvline(forward, color="black", linestyle=":", label="Forward")
    plt.title(
        f"Implied volatility skew - {first.ticker} - "
        f"{first.trade_date.date()} - maturity {first.maturity_date.date()}"
    )
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_surface_skews(
    points: list[ImpliedVolPoint],
    output_path: str | Path,
    max_maturities: int = 8,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_points = group_by_maturity(points)
    maturities = sorted(grouped_points.keys())[:max_maturities]

    plt.figure(figsize=(10, 6))
    for maturity_date in maturities:
        maturity_points = sorted(grouped_points[maturity_date], key=lambda point: point.strike)
        plt.plot(
            [point.strike for point in maturity_points],
            [point.implied_vol for point in maturity_points],
            marker="o",
            markersize=3,
            linewidth=1,
            label=str(maturity_date),
        )

    first = points[0]
    plt.axvline(first.spot, color="gray", linestyle="--", label="Spot")
    plt.title(f"Implied volatility surface skews - {first.ticker}")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



