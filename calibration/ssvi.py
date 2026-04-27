from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SSVIParameters:
    maturity_date: str
    maturity: float
    forward: float | None
    theta: float
    rho: float
    phi: float
    rmse_total_variance: float


class SSVIModel:
    @staticmethod
    def total_variance(k, theta: float, rho: float, phi: float):
        k = np.asarray(k)
        return 0.5 * theta * (
            1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
        )

    @staticmethod
    def implied_vol(k, maturity: float, theta: float, rho: float, phi: float):
        total_variance = SSVIModel.total_variance(k, theta, rho, phi)
        return np.sqrt(np.maximum(total_variance, 0) / maturity)


class SSVIVolSurface:
    def __init__(self, parameters: list[SSVIParameters]):
        if len(parameters) == 0:
            raise ValueError("La surface SSVI a besoin d'au moins une maturite")

        self.parameters = sorted(parameters, key=lambda param: param.maturity)
        self.maturities = np.array([param.maturity for param in self.parameters])
        self.forwards = np.array(
            [
                np.nan if param.forward is None else param.forward
                for param in self.parameters
            ],
            dtype=float,
        )

    @classmethod
    def from_csv(cls, file_path: str = "data/ssvi_surface_params.csv"):
        df = pd.read_csv(file_path, sep=";")
        parameters = [
            SSVIParameters(
                maturity_date=str(row["maturity_date"]),
                maturity=float(row["maturity"]),
                forward=_optional_float(row.get("forward")),
                theta=float(row["theta"]),
                rho=float(row["rho"]),
                phi=float(row["phi"]),
                rmse_total_variance=float(row["rmse_total_variance"]),
            )
            for _, row in df.iterrows()
        ]
        return cls(parameters)

    def get_total_variance(self, maturity: float, strike, forward):
        if maturity <= 0:
            return 0.0
        strike_array = np.asarray(strike, dtype=float)
        forward_array = np.asarray(forward, dtype=float)
        if np.any(strike_array <= 0) or np.any(forward_array <= 0):
            raise ValueError("strike et forward doivent etre strictement positifs")

        scalar_input = strike_array.ndim == 0 and forward_array.ndim == 0
        k = np.log(strike_array / forward_array)
        if maturity <= self.maturities[0]:
            variance = self._total_variance_for_params(self.parameters[0], k)
            return float(variance) if scalar_input else variance
        if maturity >= self.maturities[-1]:
            variance = self._total_variance_for_params(self.parameters[-1], k)
            return float(variance) if scalar_input else variance

        right_index = int(np.searchsorted(self.maturities, maturity))
        left_params = self.parameters[right_index - 1]
        right_params = self.parameters[right_index]

        left_w = self._total_variance_for_params(left_params, k)
        right_w = self._total_variance_for_params(right_params, k)
        weight = (maturity - left_params.maturity) / (
            right_params.maturity - left_params.maturity
        )
        variance = left_w + weight * (right_w - left_w)
        return float(variance) if scalar_input else variance

    def get_vol(self, maturity: float, strike, forward):
        total_variance = self.get_total_variance(maturity, strike, forward)
        vol = np.sqrt(np.maximum(total_variance, 0.0) / maturity)
        return float(vol) if np.asarray(vol).ndim == 0 else vol

    def get_forward(self, maturity: float, fallback_forward: float | None = None) -> float:
        valid = ~np.isnan(self.forwards)
        if np.any(valid):
            return float(np.interp(maturity, self.maturities[valid], self.forwards[valid]))
        if fallback_forward is None:
            raise ValueError("Aucun forward disponible pour la surface SSVI")
        return float(fallback_forward)

    @staticmethod
    def _total_variance_for_params(params: SSVIParameters, k):
        return SSVIModel.total_variance(k, params.theta, params.rho, params.phi)


class SSVICalibrator:
    def calibrate_slice(self, slice_df) -> SSVIParameters:
        df = slice_df.copy()
        df["log_moneyness"] = np.log(df["strike"] / df["forward"])
        df["total_variance"] = df["implied_vol"] ** 2 * df["maturity"]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["log_moneyness", "total_variance"])

        if len(df) < 5:
            raise ValueError("Pas assez de points pour calibrer SSVI")

        atm_row = df.loc[df["log_moneyness"].abs().idxmin()]
        theta = float(atm_row["total_variance"])

        k = df["log_moneyness"].to_numpy()
        market_w = df["total_variance"].to_numpy()

        def objective(params):
            rho, phi = params
            model_w = SSVIModel.total_variance(k, theta, rho, phi)
            return np.mean((model_w - market_w) ** 2)

        result = minimize(
            objective,
            x0=np.array([-0.4, 2.0]),
            bounds=[(-0.999, 0.999), (1e-6, 100.0)],
            method="L-BFGS-B",
        )

        if not result.success:
            raise RuntimeError(f"Calibration SSVI impossible: {result.message}")

        rho, phi = result.x
        rmse = float(np.sqrt(objective(result.x)))
        return SSVIParameters(
            maturity_date=str(df["maturity_date"].iloc[0]),
            maturity=float(df["maturity"].iloc[0]),
            forward=float(df["forward"].median()) if "forward" in df.columns else None,
            theta=theta,
            rho=float(rho),
            phi=float(phi),
            rmse_total_variance=rmse,
        )


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)



