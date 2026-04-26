from __future__ import annotations

from math import exp, log, pi

import numpy as np
from scipy.integrate import quad

from pricing.heston import HestonParameters


class HestonFourierPricer:
    def __init__(
        self,
        parameters: HestonParameters,
        integration_upper_bound: float = 100.0,
        integration_limit: int = 100,
    ):
        self.parameters = parameters
        self.integration_upper_bound = integration_upper_bound
        self.integration_limit = integration_limit

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        option_type: str,
    ) -> float:
        if maturity <= 0:
            if option_type == "call":
                return max(spot - strike, 0.0)
            if option_type == "put":
                return max(strike - spot, 0.0)
            raise ValueError("option_type doit etre 'call' ou 'put'")

        call = self.call_price(spot, strike, maturity, rate, dividend_yield)
        if option_type == "call":
            return call
        if option_type == "put":
            return call - spot * exp(-dividend_yield * maturity) + strike * exp(
                -rate * maturity
            )
        raise ValueError("option_type doit etre 'call' ou 'put'")

    def call_price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        phi_minus_i = self._characteristic_function(
            -1j,
            spot,
            maturity,
            rate,
            dividend_yield,
        )
        if abs(phi_minus_i) < 1e-12:
            raise ValueError("Caracteristique Heston instable pour u=-i")

        log_strike = log(strike)

        def p1_integrand(u: float) -> float:
            numerator = np.exp(-1j * u * log_strike) * self._characteristic_function(
                u - 1j,
                spot,
                maturity,
                rate,
                dividend_yield,
            )
            value = numerator / (1j * u * phi_minus_i)
            return float(np.real(value))

        def p2_integrand(u: float) -> float:
            numerator = np.exp(-1j * u * log_strike) * self._characteristic_function(
                u,
                spot,
                maturity,
                rate,
                dividend_yield,
            )
            value = numerator / (1j * u)
            return float(np.real(value))

        p1 = 0.5 + self._integrate(p1_integrand) / pi
        p2 = 0.5 + self._integrate(p2_integrand) / pi
        price = spot * exp(-dividend_yield * maturity) * p1 - strike * exp(
            -rate * maturity
        ) * p2
        return float(max(price, 0.0))

    def _integrate(self, integrand) -> float:
        value, _ = quad(
            integrand,
            1e-8,
            self.integration_upper_bound,
            limit=self.integration_limit,
            epsabs=1e-6,
            epsrel=1e-6,
        )
        return value

    def _characteristic_function(
        self,
        u: complex,
        spot: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> complex:
        params = self.parameters
        kappa = params.kappa
        theta = params.theta
        sigma_v = params.sigma_v
        rho = float(np.clip(params.rho, -0.999, 0.999))
        v0 = params.v0

        if sigma_v <= 1e-10:
            sigma_v = 1e-10

        iu = 1j * u
        d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v**2 * (iu + u**2))
        g = (kappa - rho * sigma_v * iu - d) / (
            kappa - rho * sigma_v * iu + d
        )
        exp_minus_dt = np.exp(-d * maturity)

        c = (
            iu * (log(spot) + (rate - dividend_yield) * maturity)
            + (kappa * theta / sigma_v**2)
            * (
                (kappa - rho * sigma_v * iu - d) * maturity
                - 2.0 * np.log((1.0 - g * exp_minus_dt) / (1.0 - g))
            )
        )
        d_term = (
            (kappa - rho * sigma_v * iu - d)
            / sigma_v**2
            * ((1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt))
        )
        return np.exp(c + d_term * v0)
