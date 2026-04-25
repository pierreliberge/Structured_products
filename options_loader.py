from datetime import datetime, timezone
from pathlib import Path
import csv

from market_data import OptionCalibrationPoint


class OptionsCSVLoader:
    def __init__(self, file_path: str | Path = "data/options.csv", delimiter: str = ";"):
        self.file_path = Path(file_path)
        self.delimiter = delimiter

    def load(
        self,
        ticker: str | None = None,
        option_type: str | None = None,
        min_price: float = 0.0,
        min_maturity: float = 0.0,
    ) -> list[OptionCalibrationPoint]:
        file_path = self._resolve_file_path()
        with file_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=self.delimiter)
            points = []
            for row in reader:
                if ticker is not None and row["ticker"] != ticker:
                    continue
                if option_type is not None and row["side"] != option_type:
                    continue

                point = self._row_to_calibration_point(row)
                if point is None:
                    continue
                if point.option_price <= min_price:
                    continue
                if point.maturity <= min_maturity:
                    continue
                points.append(point)

            return points

    def _row_to_calibration_point(
        self,
        row: dict[str, str],
    ) -> OptionCalibrationPoint | None:
        trade_date = self._date_to_datetime(row["date"])
        maturity_date = self._timestamp_to_datetime(row["expiration"])
        spot = self._to_float(row["underlyingPrice"])
        in_the_money = self._to_bool(row["inTheMoney"])

        if trade_date is None or maturity_date is None:
            return None
        if spot is None or in_the_money is None:
            return None

        option_price = self._get_option_price(row)
        if option_price is None:
            return None

        maturity = (maturity_date.date() - trade_date.date()).days / 365
        return OptionCalibrationPoint(
            trade_date=trade_date,
            underlying=row["underlying"],
            maturity_date=maturity_date,
            maturity=maturity,
            strike=self._to_float(row["strike"]) or 0.0,
            option_price=option_price,
            option_type=row["side"],
            in_the_money=in_the_money,
            spot=spot,
            ticker=row["ticker"],
            intrinsic_value=self._to_float(row["intrinsicValue"]),
            extrinsic_value=self._to_float(row["extrinsicValue"]),
        )

    @staticmethod
    def _get_option_price(row: dict[str, str]) -> float | None:
        mid = OptionsCSVLoader._to_float(row["mid"])
        return mid if mid is not None and mid > 0 else None

    def _resolve_file_path(self) -> Path:
        if self.file_path.exists():
            return self.file_path

        if self.file_path.name == "options.csv":
            singular_name = self.file_path.with_name("option.csv")
            if singular_name.exists():
                return singular_name

        data_dir = self.file_path.parent
        csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
        option_csv_files = [
            path
            for path in csv_files
            if "option" in path.name.lower() and "implied_vol" not in path.name.lower()
        ]
        if len(option_csv_files) == 1:
            return option_csv_files[0]
        if len(csv_files) == 1:
            return csv_files[0]

        raise FileNotFoundError(f"Fichier introuvable: {self.file_path}")

    @staticmethod
    def _timestamp_to_datetime(value: str) -> datetime | None:
        if not value:
            return None
        return datetime.fromtimestamp(int(value), tz=timezone.utc)

    @staticmethod
    def _date_to_datetime(value: str) -> datetime | None:
        if not value:
            return None
        return datetime.strptime(value, "%Y-%m-%d")

    @staticmethod
    def _to_float(value: str) -> float | None:
        return float(value) if value else None

    @staticmethod
    def _to_bool(value: str) -> bool | None:
        if not value:
            return None
        return value.lower() == "true"
