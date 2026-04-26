import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssvi import SSVICalibrator, SSVIModel


def select_slice(df: pd.DataFrame, maturity_date: str | None) -> pd.DataFrame:
    if maturity_date is not None:
        slice_df = df[df["maturity_date"] == maturity_date].copy()
        if slice_df.empty:
            raise ValueError(f"Aucune maturite trouvee: {maturity_date}")
        return slice_df

    counts = df["maturity_date"].value_counts()
    selected_date = counts.idxmax()
    return df[df["maturity_date"] == selected_date].copy()


def save_parameters(params, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["maturity_date", "maturity", "theta", "rho", "phi", "rmse_total_variance"])
        writer.writerow(
            [
                params.maturity_date,
                params.maturity,
                params.theta,
                params.rho,
                params.phi,
                params.rmse_total_variance,
            ]
        )


def plot_slice_fit(slice_df: pd.DataFrame, params, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = slice_df.copy()
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    df = df.sort_values("strike")

    k_grid = np.linspace(df["log_moneyness"].min(), df["log_moneyness"].max(), 250)
    strikes_grid = float(df["forward"].iloc[0]) * np.exp(k_grid)
    ssvi_iv = SSVIModel.implied_vol(
        k_grid,
        params.maturity,
        params.theta,
        params.rho,
        params.phi,
    )

    plt.figure(figsize=(9, 5))
    plt.scatter(df["strike"], df["implied_vol"], s=18, label="Market IV")
    plt.plot(strikes_grid, ssvi_iv, color="black", linewidth=2, label="SSVI fit")
    plt.axvline(float(df["forward"].iloc[0]), color="gray", linestyle=":", label="Forward")
    plt.title(f"SSVI slice fit - {df['ticker'].iloc[0]} - {params.maturity_date}")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/implied_vol_surface.csv")
    parser.add_argument("--maturity-date", default=None)
    parser.add_argument("--output-params", default="data/ssvi_slice_params.csv")
    parser.add_argument("--output-plot", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    slice_df = select_slice(df, args.maturity_date)

    calibrator = SSVICalibrator()
    params = calibrator.calibrate_slice(slice_df)

    save_parameters(params, args.output_params)
    if args.output_plot is not None:
        plot_slice_fit(slice_df, params, args.output_plot)

    print(f"SSVI calibrated for maturity {params.maturity_date}")
    print(f"theta={params.theta:.8f}, rho={params.rho:.6f}, phi={params.phi:.6f}")
    print(f"RMSE total variance={params.rmse_total_variance:.8f}")
    print(f"Parameters: {args.output_params}")
    if args.output_plot is not None:
        print(f"Plot: {args.output_plot}")


if __name__ == "__main__":
    main()
