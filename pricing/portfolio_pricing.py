from dataclasses import dataclass
from datetime import date

from core.portfolio import Portfolio, PortfolioPosition
from core.products import Option
from market.option_market_surface import CSVOptionSurfaceProvider
from pricing.engine import PricingConfig, PricingEngine


@dataclass
class PricingResult:
    portfolio: str
    line: int | None
    product: str
    underlying: str | None
    valuation_date: date
    maturity_date: date
    quantity: float
    unit_price: float | None
    market_value: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None
    dv01: float | None
    status: str
    warning: str


class PortfolioMarketPricer:
    def __init__(
        self,
        option_surface_provider: CSVOptionSurfaceProvider,
        config: PricingConfig | None = None,
    ):
        self.option_surface_provider = option_surface_provider
        self.engine = PricingEngine(option_surface_provider, config)
        self.rate_curve = self.engine.rate_curve
        self.dividend_yield = self.engine.dividend_yield
        self.config = self.engine.config

    def price_portfolio(self, portfolio: Portfolio) -> list[PricingResult]:
        results = []
        for position in portfolio.positions:
            try:
                unit_price, warning = self.price_position(position)
                risks = self.risk_position(position).scaled(position.quantity)
                warning = self.engine._join_warnings(warning, risks.warning)
                status = "OK"
            except Exception as exc:
                unit_price = None
                warning = str(exc)
                risks = None
                status = "ERROR"
            market_value = None if unit_price is None else position.quantity * unit_price
            results.append(
                PricingResult(
                    portfolio=portfolio.name,
                    line=position.line,
                    product=position.name,
                    underlying=position.underlying,
                    valuation_date=position.pricing_date,
                    maturity_date=position.product.maturity_date,
                    quantity=position.quantity,
                    unit_price=unit_price,
                    market_value=market_value,
                    delta=None if risks is None else risks.delta,
                    gamma=None if risks is None else risks.gamma,
                    vega=None if risks is None else risks.vega,
                    theta=None if risks is None else risks.theta,
                    rho=None if risks is None else risks.rho,
                    dv01=None if risks is None else risks.dv01,
                    status=status,
                    warning=warning,
                )
            )
        return results

    def price_position(self, position: PortfolioPosition) -> tuple[float, str]:
        return self.engine.price_position(position)

    def price_option_leg(
        self,
        position: PortfolioPosition,
        option: Option,
    ) -> tuple[float, str]:
        return self.engine.price_option_leg(position, option)

    def risk_position(self, position: PortfolioPosition):
        return self.engine.risk_position(position)

    def risk_option_leg(self, position: PortfolioPosition, option: Option):
        return self.engine.risk_option_leg(position, option)
