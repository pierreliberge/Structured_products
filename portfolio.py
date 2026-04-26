from dataclasses import dataclass, field


@dataclass
class PortfolioPosition:
    product: object
    quantity: float
    pricing_date: object
    name: str
    underlying: str | None = None
    line: int | None = None
    metadata: dict = field(default_factory=dict)


class Portfolio:
    def __init__(self, name: str):
        self.name = name
        self.positions: list[PortfolioPosition] = []

    def add_position(self, position: PortfolioPosition) -> None:
        self.positions.append(position)

    def __len__(self) -> int:
        return len(self.positions)

    def by_underlying(self) -> dict[str, list[PortfolioPosition]]:
        grouped = {}
        for position in self.positions:
            key = position.underlying or "NA"
            grouped.setdefault(key, []).append(position)
        return grouped
