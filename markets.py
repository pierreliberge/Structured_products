from datetime import date

class Market:
    def __init__(self, spot, rate_curve, dividend_discrete = None, q= 0.0):
        self.spot = spot
        self.rate_curve = rate_curve
        self.dividend_discrete = dividend_discrete
        self.q = q
        if self.spot <= 0:
            raise ValueError("Spot must be positive")
        if isinstance(self.dividend_discrete, list) :
            for i in self.dividend_discrete:
                if isinstance(i, tuple) and len(i) == 2:
                    if isinstance(i[0], date) and isinstance(i[1], float | int) and i[1] > 0:
                        continue
                    else:
                        raise ValueError("Problem in the format of discret dividend")
                else:
                    raise ValueError("Problem in the format of discret dividend")
        if not isinstance(self.dividend_discrete, list) and self.dividend_discrete is not None:
            raise ValueError("Problem in the format of discret dividend")
        if self.dividend_discrete is not None and self.q != 0.0:
            raise ValueError("pas possible d'avoir dividendes discrets et continus")








