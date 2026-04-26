import numpy as np
from datetime import date, timedelta
from date_utils import DayCount, ScheduleGenerator
from products import Bond, Option, OptionStrategy, ZC, BondFixe, BondFloat, SwapIRS, OptionType, ExerciseType
from scipy.stats import norm
from models import BlackScholesModel
from markets import Market
from payoffs import PayoffFactory



class Pricer:
    pass




class BondPricer(Pricer):
    def __init__(self, bond: Bond, pricing_date, rate_curve=None):
        self.bond = bond
        self.pricing_date = pricing_date
        self.rate_curve = rate_curve

    def price(self):
        if isinstance(self.bond, ZC):
            return self._zc_pricer()
        if isinstance(self.bond, BondFixe):
            return self._fixe_bond_pricer()
        if isinstance(self.bond, BondFloat):
            return self._float_bond_pricer()


    def _zc_pricer(self):
        d = DayCount(self.bond.day_count)
        T = d.year_fraction(self.pricing_date, self.bond.maturity_date)
        N= self.bond.notional
        DF = self.rate_curve.discount_factor(T)
        return N*DF

    def fixed_leg_pv(self):
        d = DayCount(self.bond.day_count)
        if self.bond.payment_dates is None:
            schedule = ScheduleGenerator(self.bond.issue_date, self.bond.maturity_date, self.bond.freq)
            dates = schedule.generate_dates()
        else:
            dates = self.bond.payment_dates
        N = self.bond.notional
        c = self.bond.coupon
        pv = 0
        previous_date = self.bond.issue_date
        for payment_date in dates:
            if payment_date > self.pricing_date:
                t = d.year_fraction(previous_date, payment_date)
                coupon = N * c * t
                actualise = d.year_fraction(self.pricing_date, payment_date)
                DF = self.rate_curve.discount_factor(actualise)
                pv += DF * coupon
            previous_date = payment_date
        return pv


    def _fixe_bond_pricer(self):
        if self.bond.maturity_date <= self.pricing_date:
            raise ValueError("Bond déjà arrivé à maturité")
        pv = self.fixed_leg_pv()
        d = DayCount(self.bond.day_count)
        N = self.bond.notional
        T_matu = d.year_fraction(self.pricing_date, self.bond.maturity_date)
        DF_matu = self.rate_curve.discount_factor(T_matu)
        pv += DF_matu * N
        return pv


    def float_leg_pv(self):
        d = DayCount(self.bond.day_count)
        if self.bond.payment_dates is None:
            schedule = ScheduleGenerator(self.bond.issue_date, self.bond.maturity_date, self.bond.freq)
            dates = schedule.generate_dates()
        else:
            dates = self.bond.payment_dates
        N = self.bond.notional
        pv = 0
        previous_date = self.bond.issue_date
        for payment_date in dates:
            if payment_date > self.pricing_date:
                if previous_date >= self.pricing_date:
                    tau = d.year_fraction(previous_date, payment_date)
                    t1 = d.year_fraction(self.pricing_date, previous_date)
                    t2 = d.year_fraction(self.pricing_date, payment_date)
                    L = self.rate_curve.forward_rate(t1, t2)
                    coupon = N * (L + self.bond.spread) * tau
                    actualise = d.year_fraction(self.pricing_date, payment_date)
                    DF = self.rate_curve.discount_factor(actualise)
                    pv += DF * coupon
                else:
                    tau = d.year_fraction(previous_date, payment_date)
                    rate = self.rate_curve.get_rate(tau)
                    coupon = N * (rate + self.bond.spread) * tau
                    actualise = d.year_fraction(self.pricing_date, payment_date)
                    DF = self.rate_curve.discount_factor(actualise)
                    pv += DF * coupon

            previous_date = payment_date

        return pv


    def _float_bond_pricer(self):
        if self.bond.maturity_date <= self.pricing_date:
            raise ValueError("Bond déjà arrivé à maturité")
        d = DayCount(self.bond.day_count)
        T_matu = d.year_fraction(self.pricing_date, self.bond.maturity_date)
        N = self.bond.notional
        DF_matu = self.rate_curve.discount_factor(T_matu)
        pv = self.float_leg_pv()
        pv += N * DF_matu
        return pv


class SwapIRSPricer(Pricer):
    def __init__(self, swap: SwapIRS, pricing_date, rate_curve=None):
        self.swap = swap
        self.pricing_date = pricing_date
        self.rate_curve = rate_curve


    def price(self):
        buy_bond = self.swap.receive_leg
        sell_bond = self.swap.pay_leg
        b = BondPricer(buy_bond, self.pricing_date, self.rate_curve)
        if isinstance(buy_bond, BondFixe):
            buy = b.fixed_leg_pv()
        elif isinstance(buy_bond, BondFloat):
            buy = b.float_leg_pv()
        else:
            raise ValueError("Type de jambe non supporté")
        s = BondPricer(sell_bond, self.pricing_date, self.rate_curve)
        if isinstance(sell_bond, BondFixe):
            sell = s.fixed_leg_pv()
        elif isinstance(sell_bond, BondFloat):
            sell = s.float_leg_pv()
        else:
            raise ValueError("Type de jambe non supporté")
        return buy - sell





class BlackScholesPricer(Pricer):
    def __init__(self, opt: Option, model: BlackScholesModel, market : Market, pricing_date):
        self.opt = opt
        self.model = model
        self.market = market
        self.pricing_date = pricing_date


    def price(self):

        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            if self.opt.option_type == OptionType.CALL:
                return max(s - k, 0)
            elif self.opt.option_type == OptionType.PUT:
                return max(k - s, 0)
            else:
                raise ValueError("Option type incorrect")

        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_q, df_r = np.exp(-q*t), self.market.rate_curve.discount_factor(t)
        if self.opt.option_type == OptionType.CALL:
            prix = s*nd1*df_q - (k*df_r*nd2)
        elif self.opt.option_type == OptionType.PUT:
            prix = (k*df_r* (1-nd2)) - s*(1-nd1)*df_q
        else:
            raise ValueError("Option type incorrect")
        return prix


    def _compute_d1_d2(self, s, k, sigma, t, r, q):
        d1 = (np.log(s / k) + (r - q + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        pdf_d1 = norm.pdf(d1)
        return d1, d2, nd1, nd2, pdf_d1


    def _get_inputs(self):
        if self.opt.exercise == ExerciseType.AMERICAN:
            raise ValueError("Le pricing n'est pas adapté aux options américaines")
        s = self.market.spot
        k = self.opt.strike
        sigma = self.model.sigma
        q = self.market.q
        d = DayCount(self.opt.day_count)
        rate = self.market.rate_curve
        t = d.year_fraction(self.pricing_date, self.opt.maturity_date)
        if sigma <= 0:
            raise ValueError("Vol must be positive")
        if t < 0:
            raise ValueError("Time to maturity must be non negative")
        if s <= 0:
            raise ValueError("Spot must be positive")

        df_r = rate.discount_factor(t)
        r = -np.log(df_r) / t
        if self.market.dividend_discrete is not None:

            liste_div = self.market.dividend_discrete
            sum_div = 0
            for i in liste_div:
                if self.pricing_date < i[0] < self.opt.maturity_date:
                    time = d.year_fraction(self.pricing_date, i[0])
                    rate_div = rate.get_rate(time)
                    sum_div += i[1]*np.exp(-rate_div*time)
            s -= sum_div
            if s <= 0:
                raise ValueError("Spot est négatif ou nul après dividendes actualisés")


        return s, k, sigma, t, r, q


    def delta(self):
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            epsilon = 10e-6 * max(s,k)
            if self.opt.option_type == OptionType.CALL:
                if abs(s-k) < epsilon:
                    return 0.5
                elif s > k:
                    return 1
                else:
                    return 0
            elif self.opt.option_type == OptionType.PUT:
                if abs(s-k) < epsilon:
                    return -0.5
                elif s > k:
                    return 0
                else:
                    return -1
            else:
                raise ValueError("Option type incorrect")


        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_q = np.exp(-q * t)
        if self.opt.option_type == OptionType.CALL:
            return df_q * nd1
        elif self.opt.option_type == OptionType.PUT:
            return df_q *(nd1 - 1)
        else:
            raise ValueError("Option type incorrect")


    def gamma(self):
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            return 0
        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_q = np.exp(-q * t)
        return df_q * pdf_d1 / (s*sigma*np.sqrt(t))


    def vega(self):
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            return 0
        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_q = np.exp(-q * t)
        return s * np.sqrt(t) * pdf_d1 * df_q


    def theta(self):
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            return 0
        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_q = np.exp(-q * t)
        df_r = self.market.rate_curve.discount_factor(t)
        if self.opt.option_type == OptionType.CALL:
            result = (-(s * df_q * pdf_d1 * sigma) / (2 * np.sqrt(t))) - (r * k * df_r * nd2) + (q * s * df_q * nd1)
            return result
        elif self.opt.option_type == OptionType.PUT:
            result = (-(s * df_q * pdf_d1 * sigma) / (2 * np.sqrt(t))) + (r * k * df_r * (1 - nd2)) - (q * s * df_q * (1- nd1))
            return result
        else:
            raise ValueError("Option type incorrect")


    def rho(self):
        s, k, sigma, t, r, q = self._get_inputs()
        if t == 0:
            return 0
        d1, d2, nd1, nd2, pdf_d1 = self._compute_d1_d2(s, k, sigma, t, r, q)
        df_r = self.market.rate_curve.discount_factor(t)
        if self.opt.option_type == OptionType.CALL:
            return k * t * df_r * nd2
        elif self.opt.option_type == OptionType.PUT:
            return -k * t * df_r * (1 - nd2)
        else:
            raise ValueError("Option type incorrect")




class OptionStrategyPricer(Pricer):
    def __init__(self, strategy: OptionStrategy, leg_pricer_factory):
        self.strategy = strategy
        self.leg_pricer_factory = leg_pricer_factory

    def price(self):
        price = 0.0
        for weight, option in self.strategy.legs:
            price += weight * self.leg_pricer_factory(option).price()
        return price




class McPricer(Pricer):
    def __init__(self, opt: Option, model, market : Market, pricing_date, n_paths, n_steps):
        self.opt = opt
        self.model = model
        self.market = market
        self.pricing_date = pricing_date
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.payoff_calc = PayoffFactory.create(self.opt)

    def price(self):
        d = DayCount(self.opt.day_count)
        t = d.year_fraction(self.pricing_date, self.opt.maturity_date)
        rate = self.market.rate_curve
        df_r = rate.discount_factor(t)
        s = self.market.spot
        q = self.market.q
        if t > 0:
            r = -np.log(df_r) / t
        else:
            r = 0
        price_matrix = self.model.simulate_paths(s, r, t, q, self.n_paths, self.n_steps)
        payoff = self.payoff_calc.compute(price_matrix)
        payoff_act = payoff * np.exp(-r * t)
        price = np.mean(payoff_act)
        return price























