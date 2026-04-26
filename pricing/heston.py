from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class HestonParameters:
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float
    source: str = "manual"

    @classmethod
    def from_csv(cls, path: str | Path) -> "HestonParameters":
        df = pd.read_csv(path, sep=";")
        if df.empty:
            raise ValueError(f"Fichier de parametres Heston vide: {path}")

        row = df.iloc[0]
        return cls(
            kappa=float(row["kappa"]),
            theta=float(row["theta"]),
            sigma_v=float(row["sigma_v"]),
            rho=float(row["rho"]),
            v0=float(row["v0"]),
            source=str(row.get("source", "csv")),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "kappa": self.kappa,
                    "theta": self.theta,
                    "sigma_v": self.sigma_v,
                    "rho": self.rho,
                    "v0": self.v0,
                    "source": self.source,
                }
            ]
        )

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, sep=";", index=False)


class HestonModel:
    def __init__(self, parameters: HestonParameters):
        self.parameters = parameters

    def simulate_paths(
        self,
        spot: float,
        rate: float,
        maturity: float,
        dividend_yield: float,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        shocks=None,
    ):
        if spot <= 0:
            raise ValueError("spot doit etre positif")
        if maturity <= 0:
            raise ValueError("maturity doit etre positive")
        if n_steps <= 0:
            raise ValueError("n_steps doit etre positif")

        if shocks is None:
            rng = np.random.default_rng(seed)
            shocks = rng.normal(0, 1, size=(n_paths, n_steps, 2))

        n_paths, n_steps, n_factors = shocks.shape
        if n_factors != 2:
            raise ValueError("Les shocks Heston doivent avoir deux facteurs")

        params = self.parameters
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        rho = float(np.clip(params.rho, -0.999, 0.999))
        rho_perp = np.sqrt(1.0 - rho**2)

        spot_paths = np.empty((n_paths, n_steps + 1))
        variance_paths = np.empty((n_paths, n_steps + 1))
        spot_paths[:, 0] = spot
        variance_paths[:, 0] = max(params.v0, 1e-10)

        z_spot = shocks[:, :, 0]
        z_independent = shocks[:, :, 1]
        z_variance = rho * z_spot + rho_perp * z_independent

        for step in range(n_steps):
            variance = np.maximum(variance_paths[:, step], 0.0)
            sqrt_variance = np.sqrt(variance)

            drift = (rate - dividend_yield - 0.5 * variance) * dt
            diffusion = sqrt_variance * sqrt_dt * z_spot[:, step]
            spot_paths[:, step + 1] = spot_paths[:, step] * np.exp(drift + diffusion)

            next_variance = (
                variance_paths[:, step]
                + params.kappa * (params.theta - variance) * dt
                + params.sigma_v * sqrt_variance * sqrt_dt * z_variance[:, step]
            )
            variance_paths[:, step + 1] = np.maximum(next_variance, 0.0)

        return spot_paths
