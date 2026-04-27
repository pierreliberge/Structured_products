from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.implied_vol import ImpliedVolCalculator
from market.market_data import ImpliedVolPoint, OptionCalibrationPoint
from market.rate_curve_loader import RateCurveParquetLoader
from market.vol_surface import keep_otm_options


class CSVOptionSurface:
    def __init__(
        self,
        points: list[ImpliedVolPoint],
        market_date: date,
        source_warning: str = "",
    ):
        if len(points) == 0:
            raise ValueError("La surface a besoin d'au moins un point de volatilite")
        self.points = sorted(points, key=lambda point: (point.maturity, point.strike))
        self.market_date = market_date
        self.source_warning = source_warning
        self.spot = float(self.points[0].spot)

    def get_vol(self, maturity: float, strike: float) -> tuple[float, str]:
        maturities = sorted({point.maturity for point in self.points})
        nearest_maturity = min(maturities, key=lambda value: abs(value - maturity))
        slice_points = [
            point for point in self.points if abs(point.maturity - nearest_maturity) < 1e-12
        ]
        strikes = np.array([point.strike for point in slice_points], dtype=float)
        vols = np.array([point.implied_vol for point in slice_points], dtype=float)
        order = np.argsort(strikes)
        strikes = strikes[order]
        vols = vols[order]

        vol = float(np.interp(strike, strikes, vols))
        warnings = []
        if self.source_warning:
            warnings.append(self.source_warning)
        if abs(nearest_maturity - maturity) > 7 / 365:
            warnings.append(
                f"vol extrapolee depuis maturite marche {nearest_maturity:.4f} an"
            )
        if strike < strikes[0] or strike > strikes[-1]:
            warnings.append(
                f"strike extrapole/borne sur [{strikes[0]:.2f}, {strikes[-1]:.2f}]"
            )
        return vol, "; ".join(warnings)


class CSVOptionSurfaceProvider:
    def __init__(
        self,
        option_csv_path: str | Path,
        rate_curve_path: str | Path,
        rate_country: str = "United States",
        dividend_yield: float = 0.0,
    ):
        self.option_csv_path = Path(option_csv_path)
        self.rate_curve = RateCurveParquetLoader(rate_curve_path).load_latest_curve(
            country=rate_country,
            compounding="continuous",
        )
        self.dividend_yield = dividend_yield
        self.option_df = pd.read_csv(self.option_csv_path, sep=";")
        if "date" not in self.option_df.columns and "trade_date" in self.option_df.columns:
            self.option_df["date"] = self.option_df["trade_date"]
        if (
            "expiration" not in self.option_df.columns
            and "maturity_date" in self.option_df.columns
        ):
            self.option_df["expiration"] = self.option_df["maturity_date"]
        self.option_df["date"] = pd.to_datetime(self.option_df["date"]).dt.date
        self.option_df["expiration_dt"] = self.option_df["expiration"].map(
            self._parse_expiration
        )
        self._surfaces: dict[tuple[str, date], CSVOptionSurface] = {}

    def surface_for(self, ticker: str, valuation_date: date) -> CSVOptionSurface:
        key = (ticker, valuation_date)
        if key in self._surfaces:
            return self._surfaces[key]

        ticker_df = self.option_df[self.option_df["ticker"] == ticker].copy()
        if ticker_df.empty:
            raise ValueError(f"Aucune option dans le CSV pour {ticker}")

        available_dates = sorted(ticker_df["date"].unique())
        past_dates = [value for value in available_dates if value <= valuation_date]
        if past_dates:
            market_date = max(past_dates)
            source_warning = ""
        else:
            market_date = min(available_dates)
            source_warning = (
                f"pas de donnees options <= {valuation_date}; "
                f"utilisation de {market_date}"
            )

        market_df = ticker_df[ticker_df["date"] == market_date]
        if "implied_vol" in market_df.columns:
            iv_points = self._iv_surface_points(market_df)
        else:
            points = [self._row_to_point(row) for _, row in market_df.iterrows()]
            valid_points = [point for point in points if point is not None]
            if len(valid_points) == 0:
                raise ValueError(f"Aucun point option valide pour {ticker} au {market_date}")

            if "bloomberg_implied_vol" in market_df.columns:
                iv_points = self._bloomberg_iv_points(market_df)
            else:
                iv_points = self._option_price_iv_points(valid_points)
        if len(iv_points) == 0:
            raise ValueError(f"Impossible de construire une surface IV pour {ticker}")

        surface = CSVOptionSurface(iv_points, market_date, source_warning)
        self._surfaces[key] = surface
        return surface

    def _iv_surface_points(self, market_df) -> list[ImpliedVolPoint]:
        iv_points = []
        for _, row in market_df.iterrows():
            try:
                maturity_date = row["expiration_dt"].to_pydatetime()
                maturity = float(row["maturity"])
                rate = self.rate_curve.get_rate(maturity)
                iv_points.append(
                    ImpliedVolPoint(
                        trade_date=datetime.strptime(str(row["date"]), "%Y-%m-%d"),
                        maturity_date=maturity_date,
                        maturity=maturity,
                        strike=float(row["strike"]),
                        spot=float(row["spot"]),
                        option_price=float(row["option_price"]),
                        option_type=str(row["option_type"]),
                        implied_vol=float(row["implied_vol"]),
                        ticker=str(row["ticker"]),
                        rate=rate,
                        dividend_yield=self.dividend_yield,
                        forward=(
                            float(row["forward"])
                            if "forward" in row and pd.notna(row["forward"])
                            else None
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
        return iv_points

    def _option_price_iv_points(self, valid_points) -> list[ImpliedVolPoint]:
        iv_points = []
        for maturity_points in self._group_by_maturity(valid_points).values():
            rate = self.rate_curve.get_rate(maturity_points[0].maturity)
            otm_points = keep_otm_options(maturity_points, rate, self.dividend_yield)
            iv_points.extend(
                ImpliedVolCalculator.compute_points(
                    otm_points,
                    rate=rate,
                    dividend_yield=self.dividend_yield,
                )
            )
        return iv_points

    @staticmethod
    def _parse_expiration(value) -> datetime:
        text = str(value)
        try:
            return pd.to_datetime(int(text), unit="s", utc=True).tz_convert(None)
        except ValueError:
            return pd.to_datetime(text).to_pydatetime()

    @staticmethod
    def _row_to_point(row) -> OptionCalibrationPoint | None:
        try:
            trade_date = datetime.strptime(str(row["date"]), "%Y-%m-%d")
            maturity_date = row["expiration_dt"].to_pydatetime()
            maturity = (maturity_date.date() - trade_date.date()).days / 365
            if maturity <= 0:
                return None
            mid = float(row["mid"])
            if mid <= 0:
                return None
            return OptionCalibrationPoint(
                trade_date=trade_date,
                underlying=str(row["underlying"]),
                maturity_date=maturity_date,
                maturity=maturity,
                strike=float(row["strike"]),
                option_price=mid,
                option_type=str(row["side"]),
                in_the_money=str(row["inTheMoney"]).lower() == "true",
                spot=float(row["underlyingPrice"]),
                ticker=str(row["ticker"]),
                intrinsic_value=float(row["intrinsicValue"]),
                extrinsic_value=float(row["extrinsicValue"]),
                forward=(
                    float(row["forward"])
                    if "forward" in row and pd.notna(row["forward"])
                    else None
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _group_by_maturity(points: list[OptionCalibrationPoint]):
        grouped = {}
        for point in points:
            grouped.setdefault(point.maturity_date.date(), []).append(point)
        return grouped

    def _bloomberg_iv_points(self, market_df) -> list[ImpliedVolPoint]:
        iv_points = []
        for _, row in market_df.iterrows():
            point = self._row_to_point(row)
            if point is None or pd.isna(row["bloomberg_implied_vol"]):
                continue
            rate = self.rate_curve.get_rate(point.maturity)
            if not keep_otm_options([point], rate, self.dividend_yield):
                continue
            iv_points.append(
                ImpliedVolPoint(
                    trade_date=point.trade_date,
                    maturity_date=point.maturity_date,
                    maturity=point.maturity,
                    strike=point.strike,
                    spot=point.spot,
                    option_price=point.option_price,
                    option_type=point.option_type,
                    implied_vol=float(row["bloomberg_implied_vol"]),
                    ticker=point.ticker,
                    rate=rate,
                    dividend_yield=self.dividend_yield,
                    forward=point.forward,
                )
            )
        return iv_points



