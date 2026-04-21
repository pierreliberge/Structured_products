from datetime import date
from products import Option, BarrierOption, OptionType, ExerciseType, BarrierDirection, BarrierKind
from date_utils import DayCount
from markets import Market
from models import RateCurve, GBMModel, BlackScholesModel
from pricers import McPricer, BlackScholesPricer



rate = RateCurve([0, 0.2, 0.5, 1, 2], [0.01, 0.02, 0.023, 0.26, 0.27], "continuous")
mkt = Market(95, rate, None, 0.01)

model = GBMModel(0.2)

vanilla = Option(date(2027, 1,1), 100, OptionType.CALL, ExerciseType.EUROPEAN, "ACT/365")

pricer = McPricer(vanilla, model, mkt, date(2026, 1, 1), 100000, 500)

price = pricer.price()
print(price)

exo = BarrierOption(date(2027, 1,1), 100, OptionType.CALL, ExerciseType.EUROPEAN, "ACT/365", 90, BarrierDirection.DOWN, BarrierKind.OUT)

pricer_2 = McPricer(exo, model, mkt, date(2026, 1, 1), 5000, 10000)
price_2 = pricer_2.price()
print(price_2)


model = BlackScholesModel(0.2)
pricer_3 = BlackScholesPricer(vanilla, model, mkt, date(2026, 1, 1))
price_3 = pricer_3.price()
print(price_3)













