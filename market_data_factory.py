from datetime import datetime
from pathlib import Path

import pandas as pd

from market_data import OptionCalibrationPoint
from markets import Market
from products import ExerciseType, Option, OptionType
from rate_curve_loader import RateCurveParquetLoader
from yfinance_options_loader import YFinanceOptionsLoader


class MarketDataFactory:
    @staticmethod
    def option_from_point(
        point: OptionCalibrationPoint,
        exercise: ExerciseType = ExerciseType.EUROPEAN,
        day_count: str = "ACT/365",
    ) -> Option:
        return Option(
            point.maturity_date.date(),
            point.strike,
            OptionType(point.option_type),
            exercise,
            day_count,
        )

    @staticmethod
    def market_from_point(
        point: OptionCalibrationPoint,
        rate_curve,
        dividend_yield: float = 0.0,
    ) -> Market:
        return Market(
            point.spot,
            rate_curve,
            q=dividend_yield,
        )

    @staticmethod
    def rate_curve_from_parquet(
        file_path: str | Path = "data/1.rate_curves.parquet",
        country: str = "United States",
        compounding: str = "continuous",
    ):
        return RateCurveParquetLoader(file_path).load_latest_curve(
            country=country,
            compounding=compounding,
        )

    @staticmethod
    def dividend_yield_from_yfinance(ticker: str) -> float:
        return YFinanceOptionsLoader.get_dividend_yield(ticker)

    @staticmethod
    def point_from_yfinance_csv_row(row) -> OptionCalibrationPoint:
        trade_date = datetime.strptime(str(row["date"]), "%Y-%m-%d")
        maturity_date = datetime.strptime(str(row["expiration"]), "%Y-%m-%d")
        return OptionCalibrationPoint(
            trade_date=trade_date,
            underlying=str(row["underlying"]),
            maturity_date=maturity_date,
            maturity=float(row["maturity"]),
            strike=float(row["strike"]),
            option_price=float(row["mid"]),
            option_type=str(row["side"]),
            in_the_money=MarketDataFactory._to_bool(row["inTheMoney"]),
            spot=float(row["underlyingPrice"]),
            ticker=str(row["ticker"]),
            intrinsic_value=MarketDataFactory._to_optional_float(row["intrinsicValue"]),
            extrinsic_value=MarketDataFactory._to_optional_float(row["extrinsicValue"]),
        )

    @staticmethod
    def points_from_yfinance_csv(file_path: str | Path) -> list[OptionCalibrationPoint]:
        df = pd.read_csv(file_path, sep=";")
        return [
            MarketDataFactory.point_from_yfinance_csv_row(row)
            for _, row in df.iterrows()
        ]

    @staticmethod
    def option_and_market_from_point(
        point: OptionCalibrationPoint,
        rate_curve,
        dividend_yield: float,
        exercise: ExerciseType = ExerciseType.EUROPEAN,
        day_count: str = "ACT/365",
    ) -> tuple[Option, Market]:
        option = MarketDataFactory.option_from_point(point, exercise, day_count)
        market = MarketDataFactory.market_from_point(point, rate_curve, dividend_yield)
        return option, market

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"

    @staticmethod
    def _to_optional_float(value) -> float | None:
        if pd.isna(value) or value == "":
            return None
        return float(value)
