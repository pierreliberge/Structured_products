import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration.ssvi import SSVICalibrator, SSVIModel


def calibrate_surface(df: pd.DataFrame):
    calibrator = SSVICalibrator()
    params_by_maturity = []
    fitted_rows = []

    for maturity_date, slice_df in sorted(df.groupby("maturity_date")):
        params = calibrator.calibrate_slice(slice_df)
        params_by_maturity.append(params)

        slice_df = slice_df.copy()
        slice_df["log_moneyness"] = np.log(slice_df["strike"] / slice_df["forward"])
        slice_df["ssvi_implied_vol"] = SSVIModel.implied_vol(
            slice_df["log_moneyness"],
            params.maturity,
            params.theta,
            params.rho,
            params.phi,
        )
        fitted_rows.append(slice_df)

    fitted_df = pd.concat(fitted_rows, ignore_index=True)
    return params_by_maturity, fitted_df


def save_parameters(params_by_maturity, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["maturity_date", "maturity", "theta", "rho", "phi", "rmse_total_variance"])
        for params in params_by_maturity:
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


def save_fitted_surface(fitted_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "trade_date",
        "maturity_date",
        "maturity",
        "strike",
        "forward",
        "log_moneyness",
        "implied_vol",
        "ssvi_implied_vol",
        "ticker",
    ]
    fitted_df[columns].to_csv(output_path, sep=";", index=False)


def plot_surface_fit(
    fitted_df: pd.DataFrame,
    output_path: str | Path,
    max_maturities: int = 8,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    maturities = sorted(fitted_df["maturity_date"].unique())[:max_maturities]

    plt.figure(figsize=(10, 6))
    for maturity_date in maturities:
        slice_df = fitted_df[fitted_df["maturity_date"] == maturity_date].sort_values("strike")
        plt.scatter(
            slice_df["strike"],
            slice_df["implied_vol"],
            s=12,
            alpha=0.45,
        )
        plt.plot(
            slice_df["strike"],
            slice_df["ssvi_implied_vol"],
            linewidth=1.8,
            label=str(maturity_date),
        )

    ticker = fitted_df["ticker"].iloc[0]
    plt.title(f"SSVI surface fit - {ticker}")
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
    parser.add_argument("--output-params", default="data/ssvi_surface_params.csv")
    parser.add_argument("--output-surface", default="data/ssvi_fitted_surface.csv")
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--max-plot-maturities", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    params_by_maturity, fitted_df = calibrate_surface(df)

    save_parameters(params_by_maturity, args.output_params)
    save_fitted_surface(fitted_df, args.output_surface)
    if args.output_plot is not None:
        plot_surface_fit(fitted_df, args.output_plot, args.max_plot_maturities)

    print(f"SSVI calibrated on {len(params_by_maturity)} maturities.")
    print(f"Parameters: {args.output_params}")
    print(f"Fitted surface: {args.output_surface}")
    if args.output_plot is not None:
        print(f"Plot: {args.output_plot}")


if __name__ == "__main__":
    main()



