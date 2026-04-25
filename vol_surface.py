from collections import defaultdict
from datetime import datetime
from math import exp

from market_data import ImpliedVolPoint, OptionCalibrationPoint


def group_by_maturity(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point.maturity_date.date()].append(point)
    return dict(grouped)


def keep_otm_options(
    points: list[OptionCalibrationPoint],
    rate: float | None = None,
    dividend_yield: float = 0.0,
) -> list[OptionCalibrationPoint]:
    def boundary(point: OptionCalibrationPoint) -> float:
        if rate is None:
            return point.spot
        return point.spot * exp((rate - dividend_yield) * point.maturity)

    return [
        point
        for point in points
        if (point.option_type == "put" and point.strike <= boundary(point))
        or (point.option_type == "call" and point.strike >= boundary(point))
    ]


def keep_latest_trade_date(points: list[OptionCalibrationPoint]) -> list[OptionCalibrationPoint]:
    latest_trade_date = max(point.trade_date.date() for point in points)
    return [point for point in points if point.trade_date.date() == latest_trade_date]


def keep_maturity(
    points: list[OptionCalibrationPoint],
    maturity_date: str | None = None,
) -> list[OptionCalibrationPoint]:
    if maturity_date is not None:
        target_date = datetime.strptime(maturity_date, "%Y-%m-%d").date()
        return [point for point in points if point.maturity_date.date() == target_date]

    grouped_points = group_by_maturity(points)
    selected_date = max(grouped_points, key=lambda key: len(grouped_points[key]))
    return grouped_points[selected_date]


def build_surface_points(
    points: list[OptionCalibrationPoint],
    rate_curve,
    dividend_yield: float,
    implied_vol_calculator,
) -> list[ImpliedVolPoint]:
    surface_points = []
    grouped_points = group_by_maturity(points)

    for _, maturity_points in sorted(grouped_points.items()):
        maturity = maturity_points[0].maturity
        rate = rate_curve.get_rate(maturity)
        otm_points = keep_otm_options(maturity_points, rate, dividend_yield)
        iv_points = implied_vol_calculator.compute_points(
            otm_points,
            rate=rate,
            dividend_yield=dividend_yield,
        )
        surface_points.extend(iv_points)

    return sorted(surface_points, key=lambda point: (point.maturity, point.strike))
