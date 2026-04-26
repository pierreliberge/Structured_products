import argparse
from pathlib import Path

import pandas as pd

from inventory_loader import InventoryExcelLoader
from option_market_surface import CSVOptionSurfaceProvider
from portfolio_pricing import PricingResult, PortfolioMarketPricer


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="data/Inventaire.xlsx")
    parser.add_argument("--options-csv", default="data/2.options (1).csv")
    parser.add_argument("--rate-curve", default="data/1.rate_curves.parquet")
    parser.add_argument("--rate-country", default="United States")
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--portfolio", default="Options")
    parser.add_argument("--output", default="data/inventory_pricing.csv")
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
    pricer = PortfolioMarketPricer(surface_provider)
    results = pricer.price_portfolio(portfolios[args.portfolio])

    save_results(results, args.output)
    print_summary(results)
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
