from pathlib import Path

import pandas as pd

from core.portfolio import Portfolio, PortfolioPosition
from core.products import (
    Autocallable,
    BarrierDirection,
    BarrierKind,
    BarrierOption,
    BondFixe,
    BondFloat,
    Butterfly,
    CallSpread,
    ExerciseType,
    Option,
    OptionType,
    PutDownIn,
    PutDownOut,
    PutSpread,
    StructuredNote,
    SwapIRS,
)


class InventoryExcelLoader:
    def __init__(self, file_path: str | Path = "data/Inventaire.xlsx"):
        self.file_path = Path(file_path)

    def load(self) -> dict[str, Portfolio]:
        return {
            "Swap": self.load_swaps(),
            "Options": self.load_options(),
            "Autocall": self.load_autocalls(),
            "Notes structurées": self.load_structured_notes(),
        }

    def load_options(self) -> Portfolio:
        df = pd.read_excel(self.file_path, sheet_name="Options")
        portfolio = Portfolio("Options")
        for index, row in df.iterrows():
            product_name = str(row["Produit"]).strip()
            maturity_date = pd.to_datetime(row["Maturité"]).date()
            pricing_date = pd.to_datetime(row["Date Valorisation"]).date()
            underlying = str(row["Sous jacent"]).strip().upper()
            product = self._option_product_from_row(row, product_name, maturity_date)
            display_name = self._option_display_name(product, product_name)
            portfolio.add_position(
                PortfolioPosition(
                    product=product,
                    quantity=float(row["Quantité"]),
                    pricing_date=pricing_date,
                    name=display_name,
                    underlying=underlying,
                    line=index + 2,
                    metadata={"source_sheet": "Options"},
                )
            )
        return portfolio

    def load_swaps(self) -> Portfolio:
        df = pd.read_excel(self.file_path, sheet_name="Swap")
        portfolio = Portfolio("Swap")
        for index, row in df.iterrows():
            pricing_date = pd.to_datetime(row["Date Valorisation"]).date()
            maturity_date = pd.to_datetime(row["Maturité"]).date()
            start_date = self._optional_date(
                row,
                ["Date Départ", "Date Depart", "Date début", "Date Debut", "Date Start"],
            )
            start_date_warning = None
            if start_date is None:
                start_date = pricing_date
                start_date_warning = (
                    "date de depart absente: supposee egale a la date de valorisation"
                )
            nominal = float(row["Nominal"])
            fixed_rate = row["Taux fixe"]
            fixed_frequency = row["Fréquence fixe"]
            float_frequency_1 = row["Taux Variable 1"]
            float_frequency_2 = row["Taux variable 2"]

            if pd.notna(fixed_rate):
                receive_leg = BondFixe(
                    maturity_date,
                    start_date,
                    nominal,
                    "ACT/365",
                    float(fixed_rate),
                    self._frequency_to_payments_per_year(fixed_frequency),
                )
                pay_leg = BondFloat(
                    maturity_date,
                    start_date,
                    nominal,
                    "ACT/365",
                    self._frequency_to_payments_per_year(float_frequency_1),
                    spread=0.0,
                )
                name = "Fixed-Float Swap"
            else:
                receive_leg = BondFloat(
                    maturity_date,
                    start_date,
                    nominal,
                    "ACT/365",
                    self._frequency_to_payments_per_year(float_frequency_1),
                    spread=0.0,
                )
                pay_leg = BondFloat(
                    maturity_date,
                    start_date,
                    nominal,
                    "ACT/365",
                    self._frequency_to_payments_per_year(float_frequency_2),
                    spread=0.0,
                )
                name = "Float-Float Swap"

            portfolio.add_position(
                PortfolioPosition(
                    product=SwapIRS(maturity_date, receive_leg, pay_leg),
                    quantity=1.0,
                    pricing_date=pricing_date,
                    name=name,
                    line=index + 2,
                    metadata={
                        "source_sheet": "Swap",
                        "currency": row["Devise"],
                        "nominal": nominal,
                        "start_date": start_date,
                        "start_date_assumption": start_date_warning,
                        "receive_leg": type(receive_leg).__name__,
                        "pay_leg": type(pay_leg).__name__,
                    },
                )
            )
        return portfolio

    def load_autocalls(self) -> Portfolio:
        df = pd.read_excel(self.file_path, sheet_name="Autocall")
        portfolio = Portfolio("Autocall")
        for product_id, product_df in df.groupby("ID Produit"):
            product_df = product_df.sort_values("Date Observation")
            first_row = product_df.iloc[0]
            maturity_date = pd.to_datetime(product_df["Date Observation"].max()).date()
            pricing_date = pd.to_datetime(first_row["Date Valorisation"]).date()
            reference_date = pd.to_datetime(first_row["Date Référence"]).date()
            observations = [
                {
                    "date": pd.to_datetime(row["Date Observation"]).date(),
                    "call_level": float(row["Niveau de rappel"]),
                    "coupon": float(row["Coupon"]),
                }
                for _, row in product_df.iterrows()
            ]
            underlying = str(first_row["Sous Jacent"]).strip().upper()
            portfolio.add_position(
                PortfolioPosition(
                    product=Autocallable(
                        maturity_date,
                        observations,
                        underlying,
                        reference_date,
                    ),
                    quantity=1.0,
                    pricing_date=pricing_date,
                    name=f"Autocall {product_id}",
                    underlying=underlying,
                    metadata={"source_sheet": "Autocall", "product_id": product_id},
                )
            )
        return portfolio

    def load_structured_notes(self) -> Portfolio:
        df = pd.read_excel(self.file_path, sheet_name="Notes structurées")
        portfolio = Portfolio("Notes structurées")
        for index, row in df.iterrows():
            maturity_date = pd.to_datetime(row["Maturité"]).date()
            pricing_date = pd.to_datetime(row["Date valorisation"]).date()
            underlying = str(row["Sous jacent"]).strip().upper()
            code_product = row["Code produit SSPA"]
            product = StructuredNote(
                maturity_date,
                code_product,
                participation=self._optional_float(row["Taux de participation"]),
                barrier_1=self._optional_float(row["Barrière 1"]),
                cap=self._optional_float(row["Cap"]),
                barrier_2=self._optional_float(row["Barrière 2"]),
            )
            portfolio.add_position(
                PortfolioPosition(
                    product=product,
                    quantity=float(row["Quantité"]),
                    pricing_date=pricing_date,
                    name=f"Note {code_product}",
                    underlying=underlying,
                    line=index + 2,
                    metadata={
                        "source_sheet": "Notes structurées",
                        "currency": row["Devise taux"],
                    },
                )
            )
        return portfolio

    def _option_product_from_row(self, row, product_name: str, maturity_date):
        exercise = ExerciseType.EUROPEAN
        day_count = "ACT/365"

        if product_name == "Call Spread":
            return CallSpread(
                maturity_date,
                float(row["Strike 1"]),
                float(row["Strike 2"]),
                exercise,
                day_count,
            )
        if product_name == "Put Spread":
            return PutSpread(
                maturity_date,
                float(row["Strike 1"]),
                float(row["Strike 2"]),
                exercise,
                day_count,
            )
        if product_name == "Butterfly":
            return Butterfly(
                maturity_date,
                float(row["Strike 1"]),
                float(row["Strike 2"]),
                float(row["Strike 3"]),
                exercise,
                day_count,
            )
        if product_name in ("Call", "Put"):
            option_type = OptionType.CALL if product_name == "Call" else OptionType.PUT
            strike = float(row["Strike 1"])
            barrier_kind = row.get("Type Barrière")
            barrier_level = row.get("Niveau Barrière")
            if pd.isna(barrier_kind) or pd.isna(barrier_level):
                return Option(maturity_date, strike, option_type, exercise, day_count)
            barrier_level = float(barrier_level)
            barrier_kind_text = str(barrier_kind).strip().upper()
            if option_type == OptionType.PUT and barrier_level < strike:
                if barrier_kind_text == "IN":
                    return PutDownIn(maturity_date, strike, exercise, day_count, barrier_level)
                if barrier_kind_text == "OUT":
                    return PutDownOut(maturity_date, strike, exercise, day_count, barrier_level)
            barrier_direction = (
                BarrierDirection.UP if barrier_level >= strike else BarrierDirection.DOWN
            )
            return BarrierOption(
                maturity_date,
                strike,
                option_type,
                exercise,
                day_count,
                barrier_level,
                barrier_direction,
                BarrierKind(barrier_kind_text.lower()),
            )

        raise ValueError(f"Produit optionnel non supporte: {product_name}")

    @staticmethod
    def _option_display_name(product, fallback_name: str) -> str:
        if isinstance(product, PutDownIn):
            return "PDI"
        if isinstance(product, PutDownOut):
            return "PDO"
        if isinstance(product, BarrierOption):
            direction = product.barrier_direction.value.upper()
            kind = product.barrier_kind.value.upper()
            option_type = product.option_type.value.capitalize()
            return f"{option_type} {direction} {kind}"
        return fallback_name

    @staticmethod
    def _frequency_to_payments_per_year(value) -> int:
        if pd.isna(value):
            raise ValueError("Frequence manquante")
        text = str(value).strip().upper()
        if text.endswith("M"):
            months = int(text[:-1])
            if months <= 0 or 12 % months != 0:
                raise ValueError(f"Frequence invalide: {value}")
            return int(12 / months)
        if text.endswith("Y"):
            years = int(text[:-1])
            if years <= 0:
                raise ValueError(f"Frequence invalide: {value}")
            return max(1, int(1 / years))
        return int(float(text))

    @staticmethod
    def _optional_float(value):
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _optional_date(row, column_names: list[str]):
        for column_name in column_names:
            if column_name in row.index and pd.notna(row[column_name]):
                return pd.to_datetime(row[column_name]).date()
        return None



