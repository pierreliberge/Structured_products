from dataclasses import asdict, dataclass

import pandas as pd

from core.portfolio import Portfolio, PortfolioPosition
from core.products import BarrierOption, Option, OptionStrategy
from pricing.portfolio_pricing import PortfolioMarketPricer, PricingResult


@dataclass
class OptionBucketRow:
    portfolio: str
    line: int | None
    product: str
    underlying: str | None
    valuation_date: object
    maturity_date: object
    option_type: str
    strike: float
    leg_weight: float
    position_quantity: float
    unit_price: float | None
    market_value: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None
    status: str
    warning: str


class PortfolioReport:
    def __init__(
        self,
        portfolio: Portfolio,
        pricing_results: list[PricingResult],
        pricer: PortfolioMarketPricer,
    ):
        self.portfolio = portfolio
        self.pricing_results = pricing_results
        self.pricer = pricer

    def positions(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(result) for result in self.pricing_results])

    def summary(self) -> pd.DataFrame:
        df = self.positions()
        if df.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "portfolio": self.portfolio.name,
                    "nb_positions": len(df),
                    "nb_ok": int((df["status"] == "OK").sum()),
                    "nb_errors": int((df["status"] == "ERROR").sum()),
                    "nb_warnings": int(df["warning"].fillna("").ne("").sum()),
                    "market_value": float(df["market_value"].fillna(0.0).sum()),
                    "delta": self._sum_nullable(df["delta"]),
                    "gamma": self._sum_nullable(df["gamma"]),
                    "vega": self._sum_nullable(df["vega"]),
                    "theta": self._sum_nullable(df["theta"]),
                    "rho": self._sum_nullable(df["rho"]),
                    "dv01": self._sum_nullable(df["dv01"]),
                }
            ]
        )

    def by_underlying(self) -> pd.DataFrame:
        df = self.positions()
        if df.empty:
            return pd.DataFrame()
        df["underlying"] = df["underlying"].fillna("NA")
        return (
            df.groupby(["portfolio", "underlying"], as_index=False)
            .agg(
                market_value=("market_value", "sum"),
                delta=("delta", self._sum_nullable),
                gamma=("gamma", self._sum_nullable),
                vega=("vega", self._sum_nullable),
                theta=("theta", self._sum_nullable),
                rho=("rho", self._sum_nullable),
                dv01=("dv01", self._sum_nullable),
                nb_positions=("product", "count"),
                nb_errors=("status", lambda values: int((values == "ERROR").sum())),
            )
            .sort_values(["portfolio", "underlying"])
        )

    def by_product(self) -> pd.DataFrame:
        df = self.positions()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby(["portfolio", "product"], as_index=False)
            .agg(
                market_value=("market_value", "sum"),
                delta=("delta", self._sum_nullable),
                gamma=("gamma", self._sum_nullable),
                vega=("vega", self._sum_nullable),
                theta=("theta", self._sum_nullable),
                rho=("rho", self._sum_nullable),
                dv01=("dv01", self._sum_nullable),
                nb_positions=("product", "count"),
                nb_errors=("status", lambda values: int((values == "ERROR").sum())),
            )
            .sort_values(["portfolio", "product"])
        )

    def option_buckets(self) -> pd.DataFrame:
        rows = []
        for position in self.portfolio.positions:
            rows.extend(self._option_bucket_rows(position))
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(row) for row in rows])
        return (
            df.groupby(
                [
                    "portfolio",
                    "underlying",
                    "valuation_date",
                    "maturity_date",
                    "option_type",
                    "strike",
                ],
                as_index=False,
            )
            .agg(
                leg_quantity=("leg_weight", "sum"),
                position_quantity=("position_quantity", "sum"),
                market_value=("market_value", "sum"),
                delta=("delta", self._sum_nullable),
                gamma=("gamma", self._sum_nullable),
                vega=("vega", self._sum_nullable),
                theta=("theta", self._sum_nullable),
                rho=("rho", self._sum_nullable),
                nb_legs=("product", "count"),
                nb_errors=("status", lambda values: int((values == "ERROR").sum())),
                warnings=("warning", self._join_warning_series),
            )
            .sort_values(["portfolio", "underlying", "maturity_date", "strike"])
        )

    def _option_bucket_rows(self, position: PortfolioPosition) -> list[OptionBucketRow]:
        product = position.product
        if isinstance(product, OptionStrategy):
            rows = []
            for weight, option in product.legs:
                rows.append(self._price_option_bucket(position, option, weight))
            return rows
        if isinstance(product, Option):
            return [self._price_option_bucket(position, product, 1.0)]
        return []

    def _price_option_bucket(
        self,
        position: PortfolioPosition,
        option: Option,
        leg_weight: float,
    ) -> OptionBucketRow:
        try:
            if isinstance(option, BarrierOption):
                unit_price, warning = self.pricer.price_position(position)
                risk = self.pricer.risk_position(position)
            else:
                unit_price, warning = self.pricer.price_option_leg(position, option)
                risk = self.pricer.risk_option_leg(position, option)
            status = "OK"
        except Exception as exc:
            unit_price = None
            warning = str(exc)
            risk = None
            status = "ERROR"

        signed_quantity = position.quantity * leg_weight
        market_value = None if unit_price is None else signed_quantity * unit_price
        scaled_risk = None if risk is None else risk.scaled(signed_quantity)
        warning = warning if risk is None else self._join_warnings(warning, risk.warning)
        return OptionBucketRow(
            portfolio=self.portfolio.name,
            line=position.line,
            product=position.name,
            underlying=position.underlying,
            valuation_date=position.pricing_date,
            maturity_date=option.maturity_date,
            option_type=option.option_type.value,
            strike=option.strike,
            leg_weight=signed_quantity,
            position_quantity=position.quantity,
            unit_price=unit_price,
            market_value=market_value,
            delta=None if scaled_risk is None else scaled_risk.delta,
            gamma=None if scaled_risk is None else scaled_risk.gamma,
            vega=None if scaled_risk is None else scaled_risk.vega,
            theta=None if scaled_risk is None else scaled_risk.theta,
            rho=None if scaled_risk is None else scaled_risk.rho,
            status=status,
            warning=warning,
        )

    @staticmethod
    def _join_warning_series(values) -> str:
        return PortfolioReport._join_warnings(*values.fillna("").tolist())

    @staticmethod
    def _sum_nullable(values):
        return values.sum(min_count=1)

    @staticmethod
    def _join_warnings(*warnings: str) -> str:
        unique = []
        for warning in warnings:
            if warning and warning not in unique:
                unique.append(warning)
        return "; ".join(unique)
