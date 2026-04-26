from core.date_utils import DayCount, ScheduleGenerator
from core.products import Bond, BondFixe, BondFloat, SwapIRS, ZC


class Pricer:
    pass


class BondPricer(Pricer):
    def __init__(self, bond: Bond, pricing_date, rate_curve):
        self.bond = bond
        self.pricing_date = pricing_date
        self.rate_curve = rate_curve

    def price(self):
        if isinstance(self.bond, ZC):
            return self._zc_pricer()
        if isinstance(self.bond, BondFixe):
            return self._fixed_bond_pricer()
        if isinstance(self.bond, BondFloat):
            return self._float_bond_pricer()
        raise ValueError(f"Type de bond non supporte: {type(self.bond).__name__}")

    def fixed_leg_pv(self):
        day_count = DayCount(self.bond.day_count)
        dates = self._payment_dates()
        pv = 0.0
        previous_date = self.bond.issue_date

        for payment_date in dates:
            if payment_date > self.pricing_date:
                accrual = day_count.year_fraction(previous_date, payment_date)
                coupon = self.bond.notional * self.bond.coupon * accrual
                discount_maturity = day_count.year_fraction(self.pricing_date, payment_date)
                pv += self.rate_curve.discount_factor(discount_maturity) * coupon
            previous_date = payment_date

        return pv

    def float_leg_pv(self):
        day_count = DayCount(self.bond.day_count)
        dates = self._payment_dates()
        pv = 0.0
        previous_date = self.bond.issue_date

        for payment_date in dates:
            if payment_date > self.pricing_date:
                accrual = day_count.year_fraction(previous_date, payment_date)
                discount_maturity = day_count.year_fraction(self.pricing_date, payment_date)

                if previous_date >= self.pricing_date:
                    forward_start = day_count.year_fraction(self.pricing_date, previous_date)
                    forward_end = day_count.year_fraction(self.pricing_date, payment_date)
                    floating_rate = self.rate_curve.forward_rate(forward_start, forward_end)
                else:
                    floating_rate = self.rate_curve.get_rate(accrual)

                coupon = self.bond.notional * (floating_rate + self.bond.spread) * accrual
                pv += self.rate_curve.discount_factor(discount_maturity) * coupon

            previous_date = payment_date

        return pv

    def _zc_pricer(self):
        day_count = DayCount(self.bond.day_count)
        maturity = day_count.year_fraction(self.pricing_date, self.bond.maturity_date)
        return self.bond.notional * self.rate_curve.discount_factor(maturity)

    def _fixed_bond_pricer(self):
        if self.bond.maturity_date <= self.pricing_date:
            raise ValueError("Bond deja arrive a maturite")
        return self.fixed_leg_pv() + self._notional_pv()

    def _float_bond_pricer(self):
        if self.bond.maturity_date <= self.pricing_date:
            raise ValueError("Bond deja arrive a maturite")
        return self.float_leg_pv() + self._notional_pv()

    def _notional_pv(self):
        day_count = DayCount(self.bond.day_count)
        maturity = day_count.year_fraction(self.pricing_date, self.bond.maturity_date)
        return self.bond.notional * self.rate_curve.discount_factor(maturity)

    def _payment_dates(self):
        if self.bond.payment_dates is not None:
            return self.bond.payment_dates
        return ScheduleGenerator(
            self.bond.issue_date,
            self.bond.maturity_date,
            self.bond.freq,
        ).generate_dates()


class SwapIRSPricer(Pricer):
    def __init__(self, swap: SwapIRS, pricing_date, rate_curve):
        self.swap = swap
        self.pricing_date = pricing_date
        self.rate_curve = rate_curve

    def price(self):
        receive_leg_pv = self._leg_pv(self.swap.receive_leg)
        pay_leg_pv = self._leg_pv(self.swap.pay_leg)
        return receive_leg_pv - pay_leg_pv

    def _leg_pv(self, leg: Bond):
        bond_pricer = BondPricer(leg, self.pricing_date, self.rate_curve)
        if isinstance(leg, BondFixe):
            return bond_pricer.fixed_leg_pv()
        if isinstance(leg, BondFloat):
            return bond_pricer.float_leg_pv()
        raise ValueError(f"Type de jambe non supporte: {type(leg).__name__}")
