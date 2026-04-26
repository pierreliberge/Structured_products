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

    def get_local_vol(self, maturity: float, strike: float) -> float:
        variance = self.get_local_variance(maturity, strike)
        return float(np.sqrt(np.clip(variance, self.min_variance, self.max_vol**2)))

    def get_local_variance(self, maturity: float, strike: float) -> float:
        if maturity <= 0:
            raise ValueError("maturity doit etre strictement positive")
        if strike <= 0:
            raise ValueError("strike doit etre strictement positif")

        dt = min(self.dt, maturity / 2)
        dk = max(strike * self.relative_dk, 1e-4)
        call = self._call_price(maturity, strike)

        if maturity - dt <= 0:
            dcdt = (self._call_price(maturity + dt, strike) - call) / dt
        else:
            dcdt = (
                self._call_price(maturity + dt, strike)
                - self._call_price(maturity - dt, strike)
            ) / (2 * dt)

        dc_dk = (
            self._call_price(maturity, strike + dk)
            - self._call_price(maturity, strike - dk)
        ) / (2 * dk)
        d2c_dk2 = (
            self._call_price(maturity, strike + dk)
            - 2 * call
            + self._call_price(maturity, strike - dk)
        ) / (dk**2)

        rate = self._zero_rate(maturity)
        q = self.dividend_yield
        numerator = dcdt + (rate - q) * strike * dc_dk + q * call
        denominator = 0.5 * strike**2 * d2c_dk2

        if denominator <= 0:
            raise ValueError("densite implicite negative ou nulle dans Dupire")

        return float(numerator / denominator)

    def _call_price(self, maturity: float, strike: float) -> float:
        if maturity <= 0:
            return max(self.spot - strike, 0.0)

        rate = self._zero_rate(maturity)
        q = self.dividend_yield
        forward = self.spot * np.exp((rate - q) * maturity)
        sigma = self.implied_vol_surface.get_vol(maturity, strike, forward)

        d1 = (
            np.log(self.spot / strike)
            + (rate - q + 0.5 * sigma**2) * maturity
        ) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        df_q = np.exp(-q * maturity)
        df_r = self.rate_curve.discount_factor(maturity)
        return self.spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)

    def _zero_rate(self, maturity: float) -> float:
        if maturity <= 0:
            return 0.0
        df = self.rate_curve.discount_factor(maturity)
        return float(-np.log(df) / maturity)
