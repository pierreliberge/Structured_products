from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp
from pathlib import Path
import argparse
import csv
import re

import pandas as pd

from market.market_data import ImpliedVolPoint, OptionCalibrationPoint


@dataclass(frozen=True)
class BloombergOptionRow:
    trade_date: datetime
    underlying: str
    maturity_date: datetime
    maturity: float
    side: str
    strike: float
    bid: float
    ask: float
    last: float | None
    mid: float
    bloomberg_implied_vol: float | None
    volume: float | None
    exercise_type: str | None
    ticker: str
    underlying_price: float
    forward: float
    in_the_money: bool
    intrinsic_value: float
    extrinsic_value: float


@dataclass(frozen=True)
class BloombergExpiryMetadata:
    maturity_date: datetime
    rate: float | None = None
    forward: float | None = None


class BloombergOptionsLoader:
    def __init__(
        self,
        file_path: str | Path = "data/Total_vol_final.xlsx",
        sheet_name: str = "mai",
        underlying: str = "TTE",
        trade_date: str | datetime = "2026-04-27",
        default_rate: float = 0.0244,
    ):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.underlying = underlying
        self.trade_date = self._parse_trade_date(trade_date)
        self.default_rate = default_rate

    def load(
        self,
        ticker: str | None = None,
        option_type: str | None = None,
        min_price: float = 0.0,
        min_maturity: float = 0.0,
    ) -> list[OptionCalibrationPoint]:
        rows = self.load_rows(
            ticker=ticker,
            option_type=option_type,
            min_price=min_price,
            min_maturity=min_maturity,
        )
        return [
            OptionCalibrationPoint(
                trade_date=row.trade_date,
                underlying=row.underlying,
                maturity_date=row.maturity_date,
                maturity=row.maturity,
                strike=row.strike,
                option_price=row.mid,
                option_type=row.side,
                in_the_money=row.in_the_money,
                spot=row.underlying_price,
                ticker=row.underlying,
                intrinsic_value=row.intrinsic_value,
                extrinsic_value=row.extrinsic_value,
                forward=row.forward,
            )
            for row in rows
        ]

    def load_rows(
        self,
        ticker: str | None = None,
        option_type: str | None = None,
        min_price: float = 0.0,
        min_maturity: float = 0.0,
    ) -> list[BloombergOptionRow]:
        if ticker is not None and ticker != self.underlying:
            return []

        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=None)
        spot = self._load_spot()
        metadata_by_expiry = self._load_expiry_metadata(df)
        dividends = self._load_dividends()
        rows: list[BloombergOptionRow] = []

        for _, row in df.iterrows():
            call_ticker = self._to_text(row.iloc[0])
            if not self._is_option_ticker(call_ticker):
                continue

            call_row = self._option_row_from_block(
                row,
                side="call",
                offset=0,
                spot=spot,
                metadata_by_expiry=metadata_by_expiry,
                dividends=dividends,
            )
            put_row = self._option_row_from_block(
                row,
                side="put",
                offset=8,
                spot=spot,
                metadata_by_expiry=metadata_by_expiry,
                dividends=dividends,
            )
            for option_row in (call_row, put_row):
                if option_row is None:
                    continue
                if option_type is not None and option_row.side != option_type:
                    continue
                if option_row.mid <= min_price:
                    continue
                if option_row.maturity <= min_maturity:
                    continue
                rows.append(option_row)

        return rows

    def load_implied_vol_points(
        self,
        ticker: str | None = None,
        option_type: str | None = None,
        min_maturity: float = 0.0,
        only_otm: bool = True,
    ) -> list[ImpliedVolPoint]:
        points = []
        for row in self.load_rows(
            ticker=ticker,
            option_type=option_type,
            min_price=0.0,
            min_maturity=min_maturity,
        ):
            if row.bloomberg_implied_vol is None:
                continue
            if only_otm and not self._is_otm_from_forward(row):
                continue
            points.append(
                ImpliedVolPoint(
                    trade_date=row.trade_date,
                    maturity_date=row.maturity_date,
                    maturity=row.maturity,
                    strike=row.strike,
                    spot=row.underlying_price,
                    option_price=row.mid,
                    option_type=row.side,
                    implied_vol=row.bloomberg_implied_vol,
                    ticker=row.underlying,
                    forward=row.forward,
                )
            )
        return points

    def export_csv(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "date",
            "underlying",
            "expiration",
            "maturity",
            "side",
            "strike",
            "bid",
            "ask",
            "last",
            "mid",
            "bloomberg_implied_vol",
            "inTheMoney",
            "underlyingPrice",
            "forward",
            "intrinsicValue",
            "extrinsicValue",
            "ticker",
            "bbg_ticker",
            "volume",
            "exerciseType",
        ]
        with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, delimiter=";", fieldnames=fieldnames)
            writer.writeheader()
            for row in self.load_rows():
                writer.writerow(
                    {
                        "date": row.trade_date.date(),
                        "underlying": row.underlying,
                        "expiration": row.maturity_date.date(),
                        "maturity": row.maturity,
                        "side": row.side,
                        "strike": row.strike,
                        "bid": row.bid,
                        "ask": row.ask,
                        "last": row.last,
                        "mid": row.mid,
                        "bloomberg_implied_vol": row.bloomberg_implied_vol,
                        "inTheMoney": row.in_the_money,
                        "underlyingPrice": row.underlying_price,
                        "forward": row.forward,
                        "intrinsicValue": row.intrinsic_value,
                        "extrinsicValue": row.extrinsic_value,
                        "ticker": row.underlying,
                        "bbg_ticker": row.ticker,
                        "volume": row.volume,
                        "exerciseType": row.exercise_type,
                    }
                )

    def _option_row_from_block(
        self,
        row,
        side: str,
        offset: int,
        spot: float,
        metadata_by_expiry: dict[datetime.date, BloombergExpiryMetadata],
        dividends: list[tuple[datetime, float]],
    ) -> BloombergOptionRow | None:
        ticker = self._to_text(row.iloc[offset])
        if not self._is_option_ticker(ticker):
            return None

        maturity_date = self._expiration_from_ticker(ticker)
        strike = self._to_float(row.iloc[offset + 1])
        bid = self._to_float(row.iloc[offset + 2])
        ask = self._to_float(row.iloc[offset + 3])
        last = self._to_float(row.iloc[offset + 4])
        implied_vol_pct = self._to_float(row.iloc[offset + 5])
        volume = self._to_float(row.iloc[offset + 6])
        exercise_type = self._to_text(row.iloc[offset + 7])
        if maturity_date is None or strike is None or bid is None or ask is None:
            return None
        if bid < 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2
        maturity = (maturity_date.date() - self.trade_date.date()).days / 365
        if maturity <= 0:
            return None
        metadata = metadata_by_expiry.get(maturity_date.date())
        forward = self._forward_for_maturity(
            spot=spot,
            maturity_date=maturity_date,
            maturity=maturity,
            metadata=metadata,
            dividends=dividends,
        )

        if side == "call":
            intrinsic = max(spot - strike, 0.0)
            in_the_money = forward > strike
        else:
            intrinsic = max(strike - spot, 0.0)
            in_the_money = strike > forward

        return BloombergOptionRow(
            trade_date=self.trade_date,
            underlying=self.underlying,
            maturity_date=maturity_date,
            maturity=maturity,
            side=side,
            strike=strike,
            bid=bid,
            ask=ask,
            last=last,
            mid=mid,
            bloomberg_implied_vol=(
                implied_vol_pct / 100 if implied_vol_pct is not None else None
            ),
            volume=volume,
            exercise_type=exercise_type,
            ticker=ticker,
            underlying_price=spot,
            forward=forward,
            in_the_money=in_the_money,
            intrinsic_value=intrinsic,
            extrinsic_value=mid - intrinsic,
        )

    def _load_expiry_metadata(
        self,
        df: pd.DataFrame,
    ) -> dict[datetime.date, BloombergExpiryMetadata]:
        metadata_by_expiry = {}
        current = None
        for _, row in df.iterrows():
            first_cell = self._to_text(row.iloc[0])
            if first_cell is None:
                current = None
                continue
            parsed = self._parse_metadata_row(first_cell)
            if parsed is not None:
                metadata_by_expiry[parsed.maturity_date.date()] = parsed
                current = parsed
                continue
            if self._is_option_ticker(first_cell):
                maturity_date = self._expiration_from_ticker(first_cell)
                if maturity_date is None:
                    continue
                if current is not None:
                    metadata_by_expiry.setdefault(maturity_date.date(), current)
                else:
                    metadata_by_expiry.setdefault(
                        maturity_date.date(),
                        BloombergExpiryMetadata(maturity_date=maturity_date),
                    )
        return metadata_by_expiry

    @staticmethod
    def _parse_metadata_row(value: str) -> BloombergExpiryMetadata | None:
        date_match = re.match(r"(\d{1,2}-[A-Za-z]{3}-\d{2})", value)
        if date_match is None:
            return None

        maturity_date = datetime.strptime(date_match.group(1), "%d-%b-%y")
        rate_match = re.search(r"\bR\s+([0-9]+(?:[,.][0-9]+)?)", value)
        forward_match = re.search(r"\bFwdI\s+([0-9]+(?:[,.][0-9]+)?)", value)
        return BloombergExpiryMetadata(
            maturity_date=maturity_date,
            rate=(
                float(rate_match.group(1).replace(",", ".")) / 100
                if rate_match
                else None
            ),
            forward=(
                float(forward_match.group(1).replace(",", "."))
                if forward_match
                else None
            ),
        )

    def _forward_for_maturity(
        self,
        spot: float,
        maturity_date: datetime,
        maturity: float,
        metadata: BloombergExpiryMetadata | None,
        dividends: list[tuple[datetime, float]],
    ) -> float:
        if metadata is not None and metadata.forward is not None:
            return metadata.forward

        rate = metadata.rate if metadata is not None and metadata.rate is not None else self.default_rate
        present_value_dividends = sum(
            amount
            * exp(
                -rate
                * ((ex_date.date() - self.trade_date.date()).days / 365)
            )
            for ex_date, amount in dividends
            if self.trade_date.date() < ex_date.date() <= maturity_date.date()
        )
        return (spot - present_value_dividends) * exp(rate * maturity)

    @staticmethod
    def _is_otm_from_forward(row: BloombergOptionRow) -> bool:
        return (row.side == "put" and row.strike <= row.forward) or (
            row.side == "call" and row.strike >= row.forward
        )

    def _load_dividends(self) -> list[tuple[datetime, float]]:
        try:
            df = pd.read_excel(self.file_path, sheet_name="BDVD")
        except ValueError:
            return []

        dividends = []
        for _, row in df.iterrows():
            ex_date = self._parse_bdvd_date(row.get("Date ex-div"))
            amount = self._to_float(row.get("Dividendes par action"))
            if ex_date is None or amount is None:
                continue
            dividends.append((ex_date, amount))
        return sorted(dividends, key=lambda item: item[0])

    def _load_spot(self) -> float:
        spot_df = pd.read_excel(self.file_path, sheet_name="spot", nrows=0)
        for column in spot_df.columns:
            match = re.search(r"spot\s*=\s*([0-9]+(?:[,.][0-9]+)?)", str(column))
            if match:
                return float(match.group(1).replace(",", "."))
        raise ValueError("Spot introuvable dans la feuille 'spot'")

    @staticmethod
    def _expiration_from_ticker(ticker: str) -> datetime | None:
        match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", ticker)
        if match is None:
            return None
        month, day, year = (int(value) for value in match.groups())
        return datetime(year=2000 + year, month=month, day=day)

    @staticmethod
    def _parse_trade_date(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d")

    @staticmethod
    def _parse_bdvd_date(value) -> datetime | None:
        if pd.isna(value):
            return None
        if isinstance(value, datetime):
            return value
        return pd.to_datetime(value, errors="coerce").to_pydatetime()

    @staticmethod
    def _is_option_ticker(value: str | None) -> bool:
        return value is not None and value.startswith("TO4 ")

    @staticmethod
    def _to_text(value) -> str | None:
        if pd.isna(value):
            return None
        return str(value).strip()

    @staticmethod
    def _to_float(value) -> float | None:
        if pd.isna(value) or value == "":
            return None
        return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/Total_vol_final.xlsx")
    parser.add_argument("--output", default="data/total_options_bloomberg.csv")
    parser.add_argument("--underlying", default="TTE")
    parser.add_argument("--trade-date", default="2026-04-27")
    args = parser.parse_args()

    loader = BloombergOptionsLoader(
        args.input,
        underlying=args.underlying,
        trade_date=args.trade_date,
    )
    loader.export_csv(args.output)
    print(f"{len(loader.load_rows())} options exportees: {args.output}")


if __name__ == "__main__":
    main()
