import numpy as np
from scipy.stats import norm

from date_utils import DayCount
from markets import Market
from products import ExerciseType, Option, OptionType
from ssvi import SSVIVolSurface


class BlackScholesSurfacePricer:
    def __init__(
        self,
        opt: Option,
        vol_surface: SSVIVolSurface,
        market: Market,
        pricing_date,
    ):
        self.opt = opt
        self.vol_surface = vol_surface
        self.market = market
        self.pricing_date = pricing_date

    def price(self) -> float:
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            return self._payoff_at_maturity(s, k)

        d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        df_q = np.exp(-q * t)
        df_r = self.market.rate_curve.discount_factor(t)

        if self.opt.option_type == OptionType.CALL:
            return s * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2)
        if self.opt.option_type == OptionType.PUT:
            return k * df_r * norm.cdf(-d2) - s * df_q * norm.cdf(-d1)
        raise ValueError("Option type incorrect")

    def implied_surface_vol(self) -> float:
        _, _, sigma, _, _, _ = self._get_inputs()
        return sigma

    def _get_inputs(self):
        if self.opt.exercise == ExerciseType.AMERICAN:
            raise ValueError("Le pricing Black-Scholes surface est europeen")

        s = self.market.spot
        k = self.opt.strike
        q = self.market.q
        day_count = DayCount(self.opt.day_count)
        t = day_count.year_fraction(self.pricing_date, self.opt.maturity_date)

        if t < 0:
            raise ValueError("Time to maturity must be non negative")
        if s <= 0:
            raise ValueError("Spot must be positive")

        if t == 0:
            return s, k, 0.0, t, 0.0, q

        df_r = self.market.rate_curve.discount_factor(t)
        r = -np.log(df_r) / t
        forward = s * np.exp((r - q) * t)
        sigma = self.vol_surface.get_vol(t, k, forward)

        if sigma <= 0:
            raise ValueError("Surface volatility must be positive")

        return s, k, sigma, t, r, q

    def _payoff_at_maturity(self, s: float, k: float) -> float:
        if self.opt.option_type == OptionType.CALL:
            return max(s - k, 0.0)
        if self.opt.option_type == OptionType.PUT:
            return max(k - s, 0.0)
        raise ValueError("Option type incorrect")
