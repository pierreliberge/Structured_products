from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import norm

from calibration.implied_vol import ImpliedVolCalculator
from calibration.ssvi import SSVIVolSurface
from core.models import RateCurve
from core.payoffs import BarrierPayoffCalculator
from core.portfolio import PortfolioPosition
from core.products import BarrierOption, Option, OptionStrategy, SwapIRS
from market.option_market_surface import CSVOptionSurfaceProvider
from pricing.heston import HestonModel, HestonParameters
from pricing.local_vol import DupireLocalVolSurface
from pricing.pricers import SwapIRSPricer


@dataclass
class RiskMetrics:
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    dv01: float | None = None
    warning: str = ""

    def scaled(self, weight: float) -> "RiskMetrics":
        return RiskMetrics(
            delta=self._scale_value(self.delta, weight),
            gamma=self._scale_value(self.gamma, weight),
            vega=self._scale_value(self.vega, weight),
            theta=self._scale_value(self.theta, weight),
            rho=self._scale_value(self.rho, weight),
            dv01=self._scale_value(self.dv01, weight),
            warning=self.warning,
        )

    def add(self, other: "RiskMetrics") -> "RiskMetrics":
        return RiskMetrics(
            delta=self._add_value(self.delta, other.delta),
            gamma=self._add_value(self.gamma, other.gamma),
            vega=self._add_value(self.vega, other.vega),
            theta=self._add_value(self.theta, other.theta),
            rho=self._add_value(self.rho, other.rho),
            dv01=self._add_value(self.dv01, other.dv01),
            warning=PricingEngine._join_warnings(self.warning, other.warning),
        )

    @staticmethod
    def _scale_value(value: float | None, weight: float) -> float | None:
        if value is None:
            return None
        return value * weight

    @staticmethod
    def _add_value(left: float | None, right: float | None) -> float | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right


@dataclass
class PricingConfig:
    vanilla_model: str = "bs"
    exotic_model: str = "gbm_mc"
    rate_model: str = "deterministic"
    day_count: str = "ACT/365"
    mc_paths: int = 30000
    mc_steps_per_year: int = 252
    min_mc_steps: int = 30
    spot_bump_relative: float = 0.01
    vol_bump_absolute: float = 0.01
    rate_bump_absolute: float = 0.0001
    theta_bump_days: int = 1
    finite_difference_seed: int = 12345
    ssvi_params_path: str | None = None
    heston_params_path: str | None = None


class PricingEngine:
    def __init__(
        self,
        option_surface_provider: CSVOptionSurfaceProvider,
        config: PricingConfig | None = None,
    ):
        self.option_surface_provider = option_surface_provider
        self.config = config or PricingConfig()
        self.rate_curve = option_surface_provider.rate_curve
        self.dividend_yield = option_surface_provider.dividend_yield

    def price_position(self, position: PortfolioPosition) -> tuple[float, str]:
        product = position.product
        if isinstance(product, OptionStrategy):
            return self._price_strategy(position)
        if isinstance(product, BarrierOption):
            return self._price_barrier(position, product)
        if isinstance(product, Option):
            return self.price_option_leg(position, product)
        if isinstance(product, SwapIRS):
            return self._price_swap(position, product)
        raise ValueError(f"Pricing non implemente pour {type(product).__name__}")

    def price_option_leg(
        self,
        position: PortfolioPosition,
        option: Option,
    ) -> tuple[float, str]:
        if self.config.vanilla_model != "bs":
            raise ValueError(f"Modele vanille non supporte: {self.config.vanilla_model}")
        return self._price_vanilla_bs(position, option)

    def risk_position(self, position: PortfolioPosition) -> RiskMetrics:
        product = position.product
        if isinstance(product, OptionStrategy):
            risks = RiskMetrics()
            for weight, option in product.legs:
                leg_risk = self.risk_option_leg(position, option).scaled(weight)
                risks = risks.add(leg_risk)
            return risks
        if isinstance(product, BarrierOption):
            return self._risk_barrier_finite_differences(position, product)
        if isinstance(product, Option):
            return self.risk_option_leg(position, product)
        if isinstance(product, SwapIRS):
            return self._risk_swap_dv01(position, product)
        return RiskMetrics(warning=f"Risques non implementes pour {type(product).__name__}")

    def risk_option_leg(
        self,
        position: PortfolioPosition,
        option: Option,
    ) -> RiskMetrics:
        if isinstance(option, BarrierOption):
            return self._risk_barrier_finite_differences(position, option)
        if self.config.vanilla_model != "bs":
            return RiskMetrics(
                warning=f"greeks non supportes pour modele vanille {self.config.vanilla_model}"
            )
        return self._risk_vanilla_bs(position, option)

    def _price_strategy(self, position: PortfolioPosition) -> tuple[float, str]:
        total = 0.0
        warnings = []
        for weight, option in position.product.legs:
            price, warning = self.price_option_leg(position, option)
            total += weight * price
            warnings.append(warning)
        return total, self._join_warnings(*warnings)

    def _price_vanilla_bs(
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
        model_warning = f"vanille pricee par {self.config.vanilla_model}"
        return price, self._join_warnings(warning, model_warning)

    def _risk_vanilla_bs(
        self,
        position: PortfolioPosition,
        option: Option,
    ) -> RiskMetrics:
        maturity = self._maturity(position.pricing_date, option.maturity_date)
        surface = self.option_surface_provider.surface_for(
            position.underlying,
            position.pricing_date,
        )
        sigma, warning = surface.get_vol(maturity, option.strike)
        rate = self.rate_curve.get_rate(maturity)
        spot = surface.spot
        strike = option.strike
        q = self.dividend_yield

        d1 = (
            np.log(spot / strike)
            + (rate - q + 0.5 * sigma**2) * maturity
        ) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)
        df_q = np.exp(-q * maturity)
        df_r = np.exp(-rate * maturity)
        pdf_d1 = norm.pdf(d1)

        if option.option_type.value == "call":
            delta = df_q * norm.cdf(d1)
            theta = (
                -spot * df_q * pdf_d1 * sigma / (2 * np.sqrt(maturity))
                - rate * strike * df_r * norm.cdf(d2)
                + q * spot * df_q * norm.cdf(d1)
            )
            rho = strike * maturity * df_r * norm.cdf(d2)
        else:
            delta = df_q * (norm.cdf(d1) - 1)
            theta = (
                -spot * df_q * pdf_d1 * sigma / (2 * np.sqrt(maturity))
                + rate * strike * df_r * norm.cdf(-d2)
                - q * spot * df_q * norm.cdf(-d1)
            )
            rho = -strike * maturity * df_r * norm.cdf(-d2)

        gamma = df_q * pdf_d1 / (spot * sigma * np.sqrt(maturity))
        vega = spot * df_q * pdf_d1 * np.sqrt(maturity)

        return RiskMetrics(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            warning=self._join_warnings(warning, "greeks analytiques bs"),
        )

    def _price_barrier(
        self,
        position: PortfolioPosition,
        option: BarrierOption,
    ) -> tuple[float, str]:
        if self.config.exotic_model not in ("gbm_mc", "local_vol_mc", "heston_mc"):
            raise ValueError(f"Modele exotique non supporte: {self.config.exotic_model}")

        maturity = self._maturity(position.pricing_date, option.maturity_date)
        surface = self.option_surface_provider.surface_for(
            position.underlying,
            position.pricing_date,
        )
        sigma, warning = surface.get_vol(maturity, option.strike)
        rate = self.rate_curve.get_rate(maturity)
        n_steps = max(
            self.config.min_mc_steps,
            int(maturity * self.config.mc_steps_per_year),
        )
        rng = np.random.default_rng(self.config.finite_difference_seed)
        z = self._mc_shocks(rng, n_steps)
        if self.config.exotic_model == "gbm_mc":
            price = self._price_barrier_from_shocks(
                option,
                surface.spot,
                rate,
                maturity,
                sigma,
                z,
            )
        elif self.config.exotic_model == "local_vol_mc":
            local_vol_surface = self._local_vol_surface(surface.spot)
            price = self._price_barrier_local_vol_from_shocks(
                option,
                surface.spot,
                rate,
                maturity,
                local_vol_surface,
                z,
            )
        else:
            price = self._price_barrier_heston_from_shocks(
                option,
                surface.spot,
                rate,
                maturity,
                self._heston_parameters(),
                z,
            )
        mc_warning = (
            f"barriere pricee par {self.config.exotic_model} "
            f"({self.config.mc_paths} paths, {n_steps} steps, "
            f"seed {self.config.finite_difference_seed})"
        )
        return price, self._join_warnings(warning, mc_warning)

    def _risk_barrier_finite_differences(
        self,
        position: PortfolioPosition,
        option: BarrierOption,
    ) -> RiskMetrics:
        if self.config.exotic_model not in ("gbm_mc", "local_vol_mc", "heston_mc"):
            return RiskMetrics(
                warning=f"greeks barriere non supportes pour modele {self.config.exotic_model}"
            )

        maturity = self._maturity(position.pricing_date, option.maturity_date)
        surface = self.option_surface_provider.surface_for(
            position.underlying,
            position.pricing_date,
        )
        sigma, warning = surface.get_vol(maturity, option.strike)
        rate = self.rate_curve.get_rate(maturity)
        n_steps = max(
            self.config.min_mc_steps,
            int(maturity * self.config.mc_steps_per_year),
        )
        rng = np.random.default_rng(self.config.finite_difference_seed)
        z = self._mc_shocks(rng, n_steps)

        spot = surface.spot
        spot_bump = max(spot * self.config.spot_bump_relative, 1e-4)
        vol_bump = self.config.vol_bump_absolute
        rate_bump = self.config.rate_bump_absolute
        theta_bump = self.config.theta_bump_days / 365

        price_fn = self._barrier_price_function(option, sigma, spot)

        base = price_fn(spot, rate, maturity, z)
        price_spot_up = price_fn(spot + spot_bump, rate, maturity, z)
        price_spot_down = price_fn(max(spot - spot_bump, 1e-8), rate, maturity, z)
        vega = None
        if self.config.exotic_model != "heston_mc":
            price_vol_up = self._barrier_price_function(option, sigma + vol_bump, spot)(
                spot, rate, maturity, z
            )
            price_vol_down = self._barrier_price_function(
                option,
                max(sigma - vol_bump, 1e-6),
                spot,
            )(spot, rate, maturity, z)
            vega = (price_vol_up - price_vol_down) / (2 * vol_bump)
        price_rate_up = price_fn(spot, rate + rate_bump, maturity, z)
        price_rate_down = price_fn(spot, rate - rate_bump, maturity, z)

        theta = None
        if maturity > theta_bump:
            price_theta = price_fn(spot, rate, maturity - theta_bump, z)
            theta = (price_theta - base) / theta_bump

        delta = (price_spot_up - price_spot_down) / (2 * spot_bump)
        gamma = (price_spot_up - 2 * base + price_spot_down) / (spot_bump**2)
        rho = (price_rate_up - price_rate_down) / (2 * rate_bump)

        fd_warning = (
            "greeks barriere/PDI/PDO par differences finies MC "
            f"{self.config.exotic_model} "
            f"({self.config.mc_paths} paths, seed {self.config.finite_difference_seed})"
        )
        return RiskMetrics(
            delta=float(delta),
            gamma=float(gamma),
            vega=None if vega is None else float(vega),
            theta=None if theta is None else float(theta),
            rho=float(rho),
            warning=self._join_warnings(
                warning,
                fd_warning,
                "vega non calculee pour heston_mc" if vega is None else "",
            ),
        )

    def _price_swap(
        self,
        position: PortfolioPosition,
        swap: SwapIRS,
    ) -> tuple[float, str]:
        if self.config.rate_model != "deterministic":
            raise ValueError(f"Modele taux non supporte: {self.config.rate_model}")

        price = SwapIRSPricer(swap, position.pricing_date, self.rate_curve).price()
        warning = "swap price comme PV(receive leg) - PV(pay leg)"
        if position.name == "Float-Float Swap":
            warning += "; support float-float via deux BondFloat"
        if position.metadata.get("start_date_assumption"):
            warning += f"; {position.metadata['start_date_assumption']}"
        return price, warning

    def _risk_swap_dv01(
        self,
        position: PortfolioPosition,
        swap: SwapIRS,
    ) -> RiskMetrics:
        if self.config.rate_model != "deterministic":
            return RiskMetrics(
                warning=f"DV01 non supporte pour modele taux {self.config.rate_model}"
            )

        base_price = SwapIRSPricer(swap, position.pricing_date, self.rate_curve).price()
        bumped_curve = self._parallel_bumped_curve(self.config.rate_bump_absolute)
        bumped_price = SwapIRSPricer(swap, position.pricing_date, bumped_curve).price()
        dv01 = bumped_price - base_price
        return RiskMetrics(
            dv01=float(dv01),
            warning="DV01 swap par bump parallele +1bp de la courbe",
        )

    def _price_barrier_from_shocks(
        self,
        option: BarrierOption,
        spot: float,
        rate: float,
        maturity: float,
        sigma: float,
        shocks,
    ) -> float:
        paths = self._gbm_paths_from_shocks(
            spot,
            rate,
            maturity,
            sigma,
            shocks,
        )
        payoff = BarrierPayoffCalculator(option).compute(paths)
        return float(np.mean(payoff) * np.exp(-rate * maturity))

    def _price_barrier_local_vol_from_shocks(
        self,
        option: BarrierOption,
        spot: float,
        rate: float,
        maturity: float,
        local_vol_surface: DupireLocalVolSurface,
        shocks,
    ) -> float:
        paths = self._local_vol_paths_from_shocks(
            spot,
            rate,
            maturity,
            local_vol_surface,
            shocks,
        )
        payoff = BarrierPayoffCalculator(option).compute(paths)
        return float(np.mean(payoff) * np.exp(-rate * maturity))

    def _price_barrier_heston_from_shocks(
        self,
        option: BarrierOption,
        spot: float,
        rate: float,
        maturity: float,
        parameters: HestonParameters,
        shocks,
    ) -> float:
        paths = HestonModel(parameters).simulate_paths(
            spot,
            rate,
            maturity,
            self.dividend_yield,
            self.config.mc_paths,
            shocks.shape[1],
            shocks=shocks,
        )
        payoff = BarrierPayoffCalculator(option).compute(paths)
        return float(np.mean(payoff) * np.exp(-rate * maturity))

    def _barrier_price_function(
        self,
        option: BarrierOption,
        sigma: float,
        reference_spot: float,
    ):
        if self.config.exotic_model == "gbm_mc":
            return lambda spot, rate, maturity, shocks: self._price_barrier_from_shocks(
                option,
                spot,
                rate,
                maturity,
                sigma,
                shocks,
            )
        if self.config.exotic_model == "heston_mc":
            parameters = self._heston_parameters()
            return lambda spot, rate, maturity, shocks: self._price_barrier_heston_from_shocks(
                option,
                spot,
                rate,
                maturity,
                parameters,
                shocks,
            )
        return lambda spot, rate, maturity, shocks: self._price_barrier_local_vol_from_shocks(
            option,
            spot,
            rate,
            maturity,
            self._local_vol_surface(reference_spot),
            shocks,
        )

    def _mc_shocks(self, rng, n_steps: int):
        if self.config.exotic_model == "heston_mc":
            return rng.normal(0, 1, size=(self.config.mc_paths, n_steps, 2))
        return rng.normal(0, 1, size=(self.config.mc_paths, n_steps))

    def _gbm_paths_from_shocks(
        self,
        spot: float,
        rate: float,
        maturity: float,
        sigma: float,
        shocks,
    ):
        if spot <= 0:
            raise ValueError("spot doit etre positif")
        if maturity <= 0:
            raise ValueError("maturity doit etre positive")
        n_paths, n_steps = shocks.shape
        dt = maturity / n_steps
        increments = (
            (rate - self.dividend_yield - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * shocks
        )
        log_paths = np.empty((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(spot)
        log_paths[:, 1:] = np.log(spot) + increments.cumsum(axis=1)
        return np.exp(log_paths)

    def _local_vol_paths_from_shocks(
        self,
        spot: float,
        rate: float,
        maturity: float,
        local_vol_surface: DupireLocalVolSurface,
        shocks,
    ):
        if spot <= 0:
            raise ValueError("spot doit etre positif")
        if maturity <= 0:
            raise ValueError("maturity doit etre positive")

        n_paths, n_steps = shocks.shape
        dt = maturity / n_steps
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = spot

        for step in range(n_steps):
            t = max(step * dt, 1e-6)
            current = paths[:, step]
            sigmas = np.array(
                [
                    local_vol_surface.get_local_vol(t, max(float(price), 1e-8))
                    for price in current
                ]
            )
            drift = (rate - self.dividend_yield - 0.5 * sigmas**2) * dt
            diffusion = sigmas * np.sqrt(dt) * shocks[:, step]
            paths[:, step + 1] = current * np.exp(drift + diffusion)

        return paths

    def _local_vol_surface(self, spot: float) -> DupireLocalVolSurface:
        if self.config.ssvi_params_path is None:
            raise ValueError(
                "ssvi_params_path requis pour exotic_model='local_vol_mc'"
            )
        ssvi_surface = SSVIVolSurface.from_csv(self.config.ssvi_params_path)
        return DupireLocalVolSurface(
            ssvi_surface,
            spot,
            self.rate_curve,
            self.dividend_yield,
        )

    def _heston_parameters(self) -> HestonParameters:
        if self.config.heston_params_path is None:
            raise ValueError(
                "heston_params_path requis pour exotic_model='heston_mc'"
            )
        return HestonParameters.from_csv(self.config.heston_params_path)

    def _parallel_bumped_curve(self, bump: float) -> RateCurve:
        return RateCurve(
            list(self.rate_curve.maturities),
            [rate + bump for rate in self.rate_curve.rates],
            self.rate_curve.compounding,
        )

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
