from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from calibration.implied_vol import ImpliedVolCalculator
from market.market_data import ImpliedVolPoint
from pricing.heston import HestonParameters
from pricing.heston_fourier import HestonFourierPricer


class HestonFourierCalibrator:
    def __init__(
        self,
        points: list[ImpliedVolPoint],
        max_points: int = 80,
        integration_upper_bound: float = 80.0,
        integration_limit: int = 80,
    ):
        valid_points = [
            point
            for point in points
            if point.maturity > 0
            and point.implied_vol > 0
            and point.spot > 0
            and point.strike > 0
            and point.option_type in ("call", "put")
        ]
        if not valid_points:
            raise ValueError("Aucun point IV valide pour calibrer Heston")

        self.points = _sample_points(valid_points, max_points)
        self.integration_upper_bound = integration_upper_bound
        self.integration_limit = integration_limit

    def calibrate(
        self,
        initial_guess: HestonParameters | None = None,
        max_iterations: int = 80,
    ) -> HestonParameters:
        if initial_guess is None:
            initial_guess = HestonIVHeuristicCalibrator(self.points).estimate()

        lower = np.array([0.01, 1e-4, 0.01, -0.95, 1e-4])
        upper = np.array([10.0, 2.0, 5.0, 0.95, 2.0])
        x0 = np.array(
            [
                initial_guess.kappa,
                initial_guess.theta,
                initial_guess.sigma_v,
                initial_guess.rho,
                initial_guess.v0,
            ],
            dtype=float,
        )
        x0 = np.clip(x0, lower + 1e-8, upper - 1e-8)

        result = least_squares(
            self._residuals,
            x0,
            bounds=(lower, upper),
            max_nfev=max_iterations,
            x_scale="jac",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        params = HestonParameters(
            kappa=float(result.x[0]),
            theta=float(result.x[1]),
            sigma_v=float(result.x[2]),
            rho=float(result.x[3]),
            v0=float(result.x[4]),
            source=f"heston_fourier_rmse={self.rmse(result.x):.8f}",
        )
        return params

    def rmse(self, x) -> float:
        residuals = self._residuals(x)
        return float(np.sqrt(np.mean(residuals**2)))

    def _residuals(self, x) -> np.ndarray:
        params = HestonParameters(
            kappa=float(x[0]),
            theta=float(x[1]),
            sigma_v=float(x[2]),
            rho=float(x[3]),
            v0=float(x[4]),
            source="calibration",
        )
        pricer = HestonFourierPricer(
            params,
            integration_upper_bound=self.integration_upper_bound,
            integration_limit=self.integration_limit,
        )

        residuals = []
        for point in self.points:
            try:
                rate = point.rate if point.rate is not None else 0.0
                dividend_yield = (
                    point.dividend_yield if point.dividend_yield is not None else 0.0
                )
                model_price = pricer.price(
                    point.spot,
                    point.strike,
                    point.maturity,
                    rate,
                    dividend_yield,
                    point.option_type,
                )
                market_price = ImpliedVolCalculator.black_scholes_price(
                    point.spot,
                    point.strike,
                    point.maturity,
                    rate,
                    point.implied_vol,
                    point.option_type,
                    dividend_yield,
                )
                residuals.append((model_price - market_price) / point.spot)
            except (FloatingPointError, OverflowError, ValueError):
                residuals.append(10.0)

        return np.array(residuals, dtype=float)


class HestonIVHeuristicCalibrator:
    def __init__(self, points: list[ImpliedVolPoint]):
        self.points = [
            point
            for point in points
            if point.maturity > 0 and point.implied_vol > 0 and point.spot > 0
        ]
        if not self.points:
            raise ValueError("Aucun point IV valide pour estimer Heston")

    def estimate(self) -> HestonParameters:
        vols = np.array([point.implied_vol for point in self.points], dtype=float)
        variances = np.clip(vols**2, 1e-6, 4.0)
        theta = float(np.clip(np.median(variances), 1e-4, 2.0))
        v0 = float(np.clip(self._short_atm_variance(), 1e-4, 2.0))
        rho = float(np.clip(self._skew_proxy(), -0.9, 0.9))
        sigma_v = float(np.clip(2.0 * np.std(vols), 0.05, 2.5))

        return HestonParameters(
            kappa=1.5,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            source="iv_heuristic",
        )

    def _short_atm_variance(self) -> float:
        scored = []
        for point in self.points:
            log_moneyness = abs(np.log(point.strike / point.spot))
            scored.append((point.maturity, log_moneyness, point.implied_vol**2))
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2]

    def _skew_proxy(self) -> float:
        slopes = []
        maturities = sorted({round(point.maturity, 8) for point in self.points})
        for maturity in maturities:
            maturity_points = [
                point for point in self.points if round(point.maturity, 8) == maturity
            ]
            if len(maturity_points) < 3:
                continue
            x = np.array(
                [np.log(point.strike / point.spot) for point in maturity_points],
                dtype=float,
            )
            y = np.array([point.implied_vol for point in maturity_points], dtype=float)
            if np.ptp(x) <= 1e-8:
                continue
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)

        if not slopes:
            return -0.5

        # Negative equity skew maps naturally to negative spot/variance correlation.
        return 4.0 * float(np.median(slopes))


def load_implied_vol_points(path: str | Path) -> list[ImpliedVolPoint]:
    df = pd.read_csv(path, sep=";")
    required = {
        "trade_date",
        "maturity_date",
        "maturity",
        "strike",
        "spot",
        "option_type",
        "implied_vol",
        "ticker",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes IV manquantes: {sorted(missing)}")

    points = []
    for _, row in df.iterrows():
        points.append(
            ImpliedVolPoint(
                trade_date=_parse_date(row["trade_date"]),
                maturity_date=_parse_date(row["maturity_date"]),
                maturity=float(row["maturity"]),
                strike=float(row["strike"]),
                spot=float(row["spot"]),
                option_price=_optional_float(row.get("option_price")) or 0.0,
                option_type=str(row["option_type"]),
                implied_vol=float(row["implied_vol"]),
                ticker=str(row["ticker"]),
                rate=_optional_float(row.get("rate")),
                dividend_yield=_optional_float(row.get("dividend_yield")),
                forward=_optional_float(row.get("forward")),
            )
        )
    return points


def _parse_date(value) -> datetime:
    return pd.to_datetime(value).to_pydatetime()


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _sample_points(
    points: list[ImpliedVolPoint],
    max_points: int,
) -> list[ImpliedVolPoint]:
    points = sorted(
        points,
        key=lambda point: (
            point.maturity,
            abs(np.log(point.strike / point.spot)),
            point.strike,
        ),
    )
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return [points[index] for index in sorted(set(indices))]
