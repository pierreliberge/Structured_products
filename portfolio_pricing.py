from dataclasses import dataclass
from datetime import date

import numpy as np

from implied_vol import ImpliedVolCalculator
from models import GBMModel
from option_market_surface import CSVOptionSurfaceProvider
from payoffs import BarrierPayoffCalculator
from portfolio import Portfolio, PortfolioPosition
from pricers import SwapIRSPricer
from products import BarrierOption, Option, OptionStrategy, SwapIRS


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
    status: str
    warning: str


class PortfolioMarketPricer:
    def __init__(self, option_surface_provider: CSVOptionSurfaceProvider):
        self.option_surface_provider = option_surface_provider
        self.rate_curve = option_surface_provider.rate_curve
        self.dividend_yield = option_surface_provider.dividend_yield

    def price_portfolio(self, portfolio: Portfolio) -> list[PricingResult]:
        results = []
        for position in portfolio.positions:
            try:
                unit_price, warning = self.price_position(position)
                status = "OK"
            except Exception as exc:
                unit_price = None
                warning = str(exc)
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
                    status=status,
                    warning=warning,
                )
            )
        return results

    def price_position(self, position: PortfolioPosition) -> tuple[float, str]:
        product = position.product
        if isinstance(product, OptionStrategy):
            return self._price_strategy(position)
        if isinstance(product, BarrierOption):
            return self._price_barrier(position, product)
        if isinstance(product, Option):
            return self._price_option(position, product)
        if isinstance(product, SwapIRS):
            return self._price_swap(position, product)
        raise ValueError(f"Pricing non implemente pour {type(product).__name__}")

    def _price_strategy(self, position: PortfolioPosition) -> tuple[float, str]:
        total = 0.0
        warnings = []
        for weight, option in position.product.legs:
            price, warning = self._price_option(position, option)
            total += weight * price
            warnings.append(warning)
        return total, self._join_warnings(*warnings)

    def _price_option(
        self,
        position: PortfolioPosition,
        option: Option,
    ) -> tuple[float, str]:
        maturity = self._maturity(position.pricing_date, option.maturity_date)
        surface = self.option_surface_provider.surface_for(
            position.underlying,
            position.pricing_date,
        )
        sigma, warning = surface.get_vol(maturity, option.strike)
        rate = self.rate_curve.get_rate(maturity)
        price = ImpliedVolCalculator.black_scholes_price(
            surface.spot,
            option.strike,
            maturity,
            rate,
            sigma,
            option.option_type.value,
            self.dividend_yield,
        )
        return price, warning

    def _price_barrier(
        self,
        position: PortfolioPosition,
        option: BarrierOption,
    ) -> tuple[float, str]:
        maturity = self._maturity(position.pricing_date, option.maturity_date)
        surface = self.option_surface_provider.surface_for(
            position.underlying,
            position.pricing_date,
        )
        sigma, warning = surface.get_vol(maturity, option.strike)
        rate = self.rate_curve.get_rate(maturity)
        paths = GBMModel(sigma).simulate_paths(
            surface.spot,
            rate,
            maturity,
            self.dividend_yield,
            n_paths=30000,
            n_steps=max(30, int(maturity * 252)),
        )
        payoff = BarrierPayoffCalculator(option).compute(paths)
        price = float(np.mean(payoff) * np.exp(-rate * maturity))
        return price, self._join_warnings(warning, "barriere pricee par Monte Carlo GBM")

    def _price_swap(
        self,
        position: PortfolioPosition,
        swap: SwapIRS,
    ) -> tuple[float, str]:
        price = SwapIRSPricer(swap, position.pricing_date, self.rate_curve).price()
        warning = "swap price comme PV(receive leg) - PV(pay leg)"
        if position.name == "Float-Float Swap":
            warning += "; support float-float via deux BondFloat"
        return price, warning

    @staticmethod
    def _maturity(pricing_date: date, maturity_date: date) -> float:
        maturity = (maturity_date - pricing_date).days / 365
        if maturity <= 0:
            raise ValueError("maturite negative ou nulle")
        return maturity

    @staticmethod
    def _join_warnings(*warnings: str) -> str:
        unique = []
        for warning in warnings:
            if warning and warning not in unique:
                unique.append(warning)
        return "; ".join(unique)
