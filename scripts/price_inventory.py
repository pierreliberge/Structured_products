import argparse
from pathlib import Path

import pandas as pd

from market.inventory_loader import InventoryExcelLoader
from market.option_market_surface import CSVOptionSurfaceProvider
from pricing.engine import PricingConfig
from pricing.portfolio_pricing import PricingResult, PortfolioMarketPricer
from pricing.reporting import PortfolioReport


def save_results(results: list[PricingResult], output_path: str | Path) -> None:
    df = pd.DataFrame([result.__dict__ for result in results])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", index=False)


def print_summary(results: list[PricingResult]) -> None:
    df = pd.DataFrame([result.__dict__ for result in results])
    print(df.to_string(index=False))
    ok_df = df[df["status"] == "OK"].copy()
    if not ok_df.empty:
        total = ok_df["market_value"].sum()
        print(f"\nTotal market value: {total:.6f}")


def save_report_tables(report: PortfolioReport, report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report.positions().to_csv(report_dir / "positions.csv", sep=";", index=False)
    report.summary().to_csv(report_dir / "summary.csv", sep=";", index=False)
    report.by_underlying().to_csv(report_dir / "by_underlying.csv", sep=";", index=False)
    report.by_product().to_csv(report_dir / "by_product.csv", sep=";", index=False)
    option_buckets = report.option_buckets()
    if not option_buckets.empty:
        option_buckets.to_csv(report_dir / "option_buckets.csv", sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="data/Inventaire.xlsx")
    parser.add_argument("--options-csv", default="data/total_iv_surface_bloomberg.csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--rate-country", default="France")
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--portfolio", default="Options")
    parser.add_argument("--output", default="data/output/inventory_pricing.csv")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--vanilla-model", choices=["bs"], default="bs")
    parser.add_argument(
        "--exotic-model",
        choices=["gbm_mc", "local_vol_mc", "heston_mc"],
        default="gbm_mc",
    )
    parser.add_argument("--rate-model", choices=["deterministic"], default="deterministic")
    parser.add_argument("--day-count", default="ACT/365")
    parser.add_argument("--mc-paths", type=int, default=30000)
    parser.add_argument("--mc-steps-per-year", type=int, default=252)
    parser.add_argument("--min-mc-steps", type=int, default=30)
    parser.add_argument("--spot-bump-relative", type=float, default=0.01)
    parser.add_argument("--vol-bump-absolute", type=float, default=0.01)
    parser.add_argument("--rate-bump-absolute", type=float, default=0.0001)
    parser.add_argument("--theta-bump-days", type=int, default=1)
    parser.add_argument("--finite-difference-seed", type=int, default=12345)
    parser.add_argument("--ssvi-params", default="data/output/total_ssvi_params.csv")
    parser.add_argument("--heston-params", default="data/output/total_heston_params.csv")
    args = parser.parse_args()

    portfolios = InventoryExcelLoader(args.inventory).load()
    if args.portfolio not in portfolios:
        raise ValueError(f"Portefeuille inconnu: {args.portfolio}")

    surface_provider = CSVOptionSurfaceProvider(
        args.options_csv,
        args.rate_curve,
        args.rate_country,
        args.dividend_yield,
    )
    config = PricingConfig(
        vanilla_model=args.vanilla_model,
        exotic_model=args.exotic_model,
        rate_model=args.rate_model,
        day_count=args.day_count,
        mc_paths=args.mc_paths,
        mc_steps_per_year=args.mc_steps_per_year,
        min_mc_steps=args.min_mc_steps,
        spot_bump_relative=args.spot_bump_relative,
        vol_bump_absolute=args.vol_bump_absolute,
        rate_bump_absolute=args.rate_bump_absolute,
        theta_bump_days=args.theta_bump_days,
        finite_difference_seed=args.finite_difference_seed,
        ssvi_params_path=args.ssvi_params,
        heston_params_path=args.heston_params,
    )
    pricer = PortfolioMarketPricer(surface_provider, config)
    portfolio = portfolios[args.portfolio]
    results = pricer.price_portfolio(portfolio)

    save_results(results, args.output)
    if args.report_dir is not None:
        save_report_tables(PortfolioReport(portfolio, results, pricer), args.report_dir)
    print_summary(results)
    print(f"\nOutput: {args.output}")
    if args.report_dir is not None:
        print(f"Report dir: {args.report_dir}")


if __name__ == "__main__":
    main()



