import numpy as np
from scipy.stats import norm

from calibration.ssvi import SSVIVolSurface


class DupireLocalVolSurface:
    def __init__(
        self,
        implied_vol_surface: SSVIVolSurface,
        spot: float,
        rate_curve,
        dividend_yield: float = 0.0,
        dt: float = 3 / 365,
        relative_dk: float = 0.01,
        min_variance: float = 1e-8,
        max_vol: float = 5.0,
    ):
        self.implied_vol_surface = implied_vol_surface
        self.spot = spot
        self.rate_curve = rate_curve
        self.dividend_yield = dividend_yield
        self.dt = dt
        self.relative_dk = relative_dk
        self.min_variance = min_variance
        self.max_vol = max_vol

    def get_local_vol(self, maturity: float, strike):
        variance = self.get_local_variance(maturity, strike)
        vol = np.sqrt(np.clip(variance, self.min_variance, self.max_vol**2))
        return float(vol) if np.asarray(vol).ndim == 0 else vol

    def get_local_variance(self, maturity: float, strike):
        if maturity <= 0:
            raise ValueError("maturity doit etre strictement positive")
        strike_array = np.asarray(strike, dtype=float)
        if np.any(strike_array <= 0):
            raise ValueError("strike doit etre strictement positif")

        scalar_input = strike_array.ndim == 0
        dt = min(self.dt, maturity / 2)
        dk = np.maximum(strike_array * self.relative_dk, 1e-4)
        call = self._call_price(maturity, strike_array)

        if maturity - dt <= 0:
            dcdt = (self._call_price(maturity + dt, strike_array) - call) / dt
        else:
            dcdt = (
                self._call_price(maturity + dt, strike_array)
                - self._call_price(maturity - dt, strike_array)
            ) / (2 * dt)

        dc_dk = (
            self._call_price(maturity, strike_array + dk)
            - self._call_price(maturity, strike_array - dk)
        ) / (2 * dk)
        d2c_dk2 = (
            self._call_price(maturity, strike_array + dk)
            - 2 * call
            + self._call_price(maturity, strike_array - dk)
        ) / (dk**2)

        rate = self._zero_rate(maturity)
        q = self.dividend_yield
        numerator = dcdt + (rate - q) * strike_array * dc_dk + q * call
        denominator = 0.5 * strike_array**2 * d2c_dk2

        if np.any(denominator <= 0):
            raise ValueError("densite implicite negative ou nulle dans Dupire")

        variance = numerator / denominator
        return float(variance) if scalar_input else variance

    def _call_price(self, maturity: float, strike):
        strike_array = np.asarray(strike, dtype=float)
        scalar_input = strike_array.ndim == 0
        if maturity <= 0:
            payoff = np.maximum(self.spot - strike_array, 0.0)
            return float(payoff) if scalar_input else payoff

        rate = self._zero_rate(maturity)
        q = self.dividend_yield
        fallback_forward = self.spot * np.exp((rate - q) * maturity)
        forward = self.implied_vol_surface.get_forward(maturity, fallback_forward)
        sigma = self.implied_vol_surface.get_vol(maturity, strike_array, forward)

        d1 = (
            np.log(self.spot / strike_array)
            + (rate - q + 0.5 * sigma**2) * maturity
        ) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        df_q = np.exp(-q * maturity)
        df_r = self.rate_curve.discount_factor(maturity)
        price = self.spot * df_q * norm.cdf(d1) - strike_array * df_r * norm.cdf(d2)
        return float(price) if scalar_input else price

    def _zero_rate(self, maturity: float) -> float:
        if maturity <= 0:
            return 0.0
        df = self.rate_curve.discount_factor(maturity)
        return float(-np.log(df) / maturity)
