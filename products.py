from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseType(Enum):
    EUROPEAN = "eu"
    AMERICAN = "us"

class BarrierDirection(Enum):
    UP = "up"
    DOWN = "down"

class BarrierKind(Enum):
    IN = "in"
    OUT = "out"



class Produit:
    def __init__(self, maturity_date):
        self.maturity_date = maturity_date




class Option(Produit):
    def __init__(self, maturity_date, strike: float, option_type: OptionType, exercise: ExerciseType, day_count):
        super().__init__(maturity_date)
        self.strike = strike
        self.option_type = option_type
        self.exercise = exercise
        self.day_count = day_count
        if self.strike <= 0:
            raise ValueError("Strike must be positive")
    """
    args:
    maturity_date
    strike
    option_type: OptionType
    exercise: ExerciseType
    day_count
    
    
    """



class BarrierOption(Option):
    def __init__(self, maturity_date, strike: float, option_type: OptionType, exercise: ExerciseType, day_count, barrier_level, barrier_direction: BarrierDirection, barrier_kind: BarrierKind):
        super().__init__(maturity_date, strike, option_type, exercise, day_count)
        self.barrier_level = barrier_level
        self.barrier_direction = barrier_direction
        self.barrier_kind = barrier_kind
        if self.barrier_level <= 0:
            raise ValueError("Barrier must be positive")




class Bond(Produit):
    def __init__(self, maturity_date, issue_date, notional, day_count):
        super().__init__(maturity_date)
        self.issue_date = issue_date
        self.notional = notional
        self.day_count = day_count


class ZC(Bond):
    def __init__(self, maturity_date, issue_date, notional, day_count):
        super().__init__(maturity_date, issue_date, notional, day_count)

class BondFixe(Bond):
    def __init__(self, maturity_date, issue_date, notional, day_count, coupon, freq, payment_dates=None):
        super().__init__(maturity_date, issue_date, notional, day_count)
        self.coupon = coupon
        self.freq = freq
        self.payment_dates = payment_dates


class BondFloat(Bond):
    def __init__(self, maturity_date, issue_date, notional, day_count, freq, spread, payment_dates=None):
        super().__init__(maturity_date, issue_date, notional, day_count)
        self.freq = freq
        self.spread = spread
        self.payment_dates = payment_dates


class SwapIRS(Produit):
    def __init__(self, maturity_date, receive_leg: Bond, pay_leg: Bond):
        super().__init__(maturity_date)
        self.receive_leg = receive_leg
        self.pay_leg = pay_leg