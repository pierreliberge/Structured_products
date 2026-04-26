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


class PutDownIn(BarrierOption):
    def __init__(self, maturity_date, strike: float, exercise: ExerciseType, day_count, barrier_level):
        super().__init__(
            maturity_date,
            strike,
            OptionType.PUT,
            exercise,
            day_count,
            barrier_level,
            BarrierDirection.DOWN,
            BarrierKind.IN,
        )


class PutDownOut(BarrierOption):
    def __init__(self, maturity_date, strike: float, exercise: ExerciseType, day_count, barrier_level):
        super().__init__(
            maturity_date,
            strike,
            OptionType.PUT,
            exercise,
            day_count,
            barrier_level,
            BarrierDirection.DOWN,
            BarrierKind.OUT,
        )


class OptionStrategy(Produit):
    def __init__(self, maturity_date, legs: list[tuple[float, Option]], name: str):
        super().__init__(maturity_date)
        self.legs = legs
        self.name = name
        if len(self.legs) == 0:
            raise ValueError("Une strategie optionnelle doit avoir au moins une jambe")


class CallSpread(OptionStrategy):
    def __init__(self, maturity_date, strike_1: float, strike_2: float, exercise: ExerciseType, day_count):
        legs = [
            (1.0, Option(maturity_date, strike_1, OptionType.CALL, exercise, day_count)),
            (-1.0, Option(maturity_date, strike_2, OptionType.CALL, exercise, day_count)),
        ]
        super().__init__(maturity_date, legs, "Call Spread")


class PutSpread(OptionStrategy):
    def __init__(self, maturity_date, strike_1: float, strike_2: float, exercise: ExerciseType, day_count):
        legs = [
            (-1.0, Option(maturity_date, strike_1, OptionType.PUT, exercise, day_count)),
            (1.0, Option(maturity_date, strike_2, OptionType.PUT, exercise, day_count)),
        ]
        super().__init__(maturity_date, legs, "Put Spread")


class Butterfly(OptionStrategy):
    def __init__(self, maturity_date, strike_1: float, strike_2: float, strike_3: float, exercise: ExerciseType, day_count):
        legs = [
            (1.0, Option(maturity_date, strike_1, OptionType.CALL, exercise, day_count)),
            (-2.0, Option(maturity_date, strike_2, OptionType.CALL, exercise, day_count)),
            (1.0, Option(maturity_date, strike_3, OptionType.CALL, exercise, day_count)),
        ]
        super().__init__(maturity_date, legs, "Butterfly")


class Autocallable(Produit):
    def __init__(self, maturity_date, observations: list[dict], underlying: str, reference_date):
        super().__init__(maturity_date)
        self.observations = observations
        self.underlying = underlying
        self.reference_date = reference_date


class StructuredNote(Produit):
    def __init__(self, maturity_date, code_product, participation=None, barrier_1=None, cap=None, barrier_2=None):
        super().__init__(maturity_date)
        self.code_product = code_product
        self.participation = participation
        self.barrier_1 = barrier_1
        self.cap = cap
        self.barrier_2 = barrier_2




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



