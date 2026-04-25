from datetime import datetime
from pathlib import Path
import csv

from market_data import OptionCalibrationPoint


class YFinanceOptionsLoader:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

    def load(
        self,
        min_maturity_days: int = 30,
        max_maturity_days: int | None = None,
        max_expirations: int | None = None,
        option_type: str | None = None,
        min_price: float = 0.0,
    ) -> list[OptionCalibrationPoint]:
        yf = self._import_yfinance()
        ticker_obj = yf.Ticker(self.ticker)
        spot = self._get_spot(ticker_obj)
        trade_date = datetime.today()

        expirations = list(ticker_obj.options)
        if max_expirations is not None:
            expirations = expirations[:max_expirations]

        points = []
        for expiration in expirations:
            maturity_date = datetime.strptime(expiration, "%Y-%m-%d")
            maturity_days = (maturity_date.date() - trade_date.date()).days

            if maturity_days < min_maturity_days:
                continue
            if max_maturity_days is not None and maturity_days > max_maturity_days:
                continue

            chain = ticker_obj.option_chain(expiration)
            if option_type in (None, "call"):
                points.extend(
                    self._rows_to_points(
                        chain.calls,
                        trade_date,
                        maturity_date,
                        spot,
                        "call",
                        min_price,
                    )
                )
            if option_type in (None, "put"):
                points.extend(
                    self._rows_to_points(
                        chain.puts,
                        trade_date,
                        maturity_date,
                        spot,
                        "put",
                        min_price,
                    )
                )

        return points

    def _rows_to_points(
        self,
        rows,
        trade_date: datetime,
        maturity_date: datetime,
        spot: float,
        option_type: str,
        min_price: float,
    ) -> list[OptionCalibrationPoint]:
        points = []
        maturity = (maturity_date.date() - trade_date.date()).days / 365

        for _, row in rows.iterrows():
            bid = self._to_float(row.get("bid"))
            ask = self._to_float(row.get("ask"))
            strike = self._to_float(row.get("strike"))

            if bid is None or ask is None or strike is None:
                continue
            if bid <= 0 or ask <= 0:
                continue

            option_price = (bid + ask) / 2
            if option_price <= min_price:
                continue

            intrinsic_value = self._intrinsic_value(spot, strike, option_type)
            points.append(
                OptionCalibrationPoint(
                    trade_date=trade_date,
                    underlying=self.ticker,
                    maturity_date=maturity_date,
                    maturity=maturity,
                    strike=strike,
                    option_price=option_price,
                    option_type=option_type,
                    in_the_money=self._is_in_the_money(spot, strike, option_type),
                    spot=spot,
                    ticker=self.ticker,
                    intrinsic_value=intrinsic_value,
                    extrinsic_value=option_price - intrinsic_value,
                )
            )

        return points

    @staticmethod
    def _import_yfinance():
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance n'est pas installe. Installe-le avec: py -m pip install yfinance"
            ) from exc
        return yf

    @staticmethod
    def _get_spot(ticker_obj) -> float:
        fast_info = getattr(ticker_obj, "fast_info", {})
        spot = fast_info.get("last_price") if fast_info else None
        if spot is not None:
            return float(spot)

        history = ticker_obj.history(period="1d")
        if history.empty:
            raise ValueError("Impossible de recuperer le prix spot via yfinance")
        return float(history["Close"].iloc[-1])

    @staticmethod
    def get_dividend_yield(ticker: str) -> float:
        yf = YFinanceOptionsLoader._import_yfinance()
        ticker_obj = yf.Ticker(ticker)

        info = ticker_obj.info
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            return YFinanceOptionsLoader._normalize_yield(float(dividend_yield))

        trailing_yield = info.get("trailingAnnualDividendYield")
        if trailing_yield is not None:
            return YFinanceOptionsLoader._normalize_yield(float(trailing_yield))

        spot = YFinanceOptionsLoader._get_spot(ticker_obj)
        dividends = ticker_obj.get_dividends(period="1y")
        if dividends.empty:
            return 0.0

        return float(dividends.sum()) / spot

    @staticmethod
    def _normalize_yield(value: float) -> float:
        if value > 0.2:
            return value / 100
        return value

    @staticmethod
    def _intrinsic_value(spot: float, strike: float, option_type: str) -> float:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    @staticmethod
    def _is_in_the_money(spot: float, strike: float, option_type: str) -> bool:
        if option_type == "call":
            return strike < spot
        return strike > spot

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            if value != value:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


def save_calibration_points(
    points: list[OptionCalibrationPoint],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(
            [
                "date",
                "underlying",
                "expiration",
                "maturity",
                "side",
                "strike",
                "mid",
                "inTheMoney",
                "underlyingPrice",
                "intrinsicValue",
                "extrinsicValue",
                "ticker",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.trade_date.date(),
                    point.underlying,
                    point.maturity_date.date(),
                    point.maturity,
                    point.option_type,
                    point.strike,
                    point.option_price,
                    point.in_the_money,
                    point.spot,
                    point.intrinsic_value,
                    point.extrinsic_value,
                    point.ticker,
                ]
            )
