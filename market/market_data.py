from dataclasses import dataclass
from datetime import datetime


@dataclass
class OptionCalibrationPoint:
    trade_date: datetime
    underlying: str
    maturity_date: datetime
    maturity: float
    strike: float
    option_price: float
    option_type: str
    in_the_money: bool
    spot: float
    ticker: str
    intrinsic_value: float | None = None
    extrinsic_value: float | None = None


@dataclass
class ImpliedVolPoint:
    trade_date: datetime
    maturity_date: datetime
    maturity: float
    strike: float
    spot: float
    option_price: float
    option_type: str
    implied_vol: float
    ticker: str
    rate: float | None = None
    dividend_yield: float | None = None
    forward: float | None = None



