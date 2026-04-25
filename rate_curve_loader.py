from pathlib import Path
import re

import pandas as pd

from models import RateCurve


class RateCurveParquetLoader:
    def __init__(self, file_path: str | Path = "data/1.rate_curves.parquet"):
        self.file_path = Path(file_path)

    def load_latest_curve(
        self,
        country: str = "United States",
        compounding: str = "continuous",
    ) -> RateCurve:
        df = pd.read_parquet(self.file_path)
        df = df[df["country"] == country].copy()
        if df.empty:
            raise ValueError(f"Aucune courbe trouvee pour le pays: {country}")

        df["date"] = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        curve_df = df[df["date"] == latest_date].copy()
        if curve_df.empty:
            raise ValueError(f"Aucune courbe disponible a la date: {latest_date}")

        curve_df["maturity_years"] = curve_df["maturity"].map(self._maturity_to_years)
        curve_df = curve_df.dropna(subset=["maturity_years", "rate"])
        curve_df = curve_df.sort_values("maturity_years")

        maturities = curve_df["maturity_years"].tolist()
        rates = (curve_df["rate"] / 100).tolist()
        return RateCurve(maturities, rates, compounding)

    @staticmethod
    def _maturity_to_years(maturity: str) -> float | None:
        match = re.fullmatch(r"(\d+)([MY])", maturity)
        if match is None:
            return None

        value = int(match.group(1))
        unit = match.group(2)
        if unit == "M":
            return value / 12
        if unit == "Y":
            return float(value)
        return None
