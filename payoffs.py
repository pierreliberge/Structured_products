from products import Option, BarrierOption, OptionType, BarrierKind, BarrierDirection
from abc import ABC, abstractmethod
import numpy as np



class PayoffCalculator(ABC):
    @abstractmethod
    def compute(self, price_matrix):
        pass


class BarrierPayoffCalculator(PayoffCalculator):
    def __init__(self, opt: BarrierOption):
        self.opt = opt
    def compute(self, price_matrix):
        k = self.opt.strike
        b = self.opt.barrier_level
        direction = self.opt.barrier_direction
        kind = self.opt.barrier_kind
        if direction == BarrierDirection.UP:
            max_by_path = np.max(price_matrix[:, 1:], axis=1)
            masque = max_by_path >= b
        else:
            min_by_path = np.min(price_matrix[:, 1:], axis=1)
            masque = min_by_path <= b
        if kind == BarrierKind.OUT:
            masque = ~masque
        if self.opt.option_type == OptionType.CALL:
            payoff = np.maximum(price_matrix[:, -1] - k, 0)
        elif self.opt.option_type == OptionType.PUT:
            payoff = np.maximum(k - price_matrix[:, -1], 0)
        else:
            raise ValueError("Option type incorrect")
        return payoff * masque



class VanillaPayoffCalculator(PayoffCalculator):
    def __init__(self, opt: Option):
        self.opt = opt
    def compute(self, price_matrix):
        k = self.opt.strike
        if self.opt.option_type == OptionType.CALL:
            payoff = np.maximum(price_matrix[:, -1] - k, 0)
        elif self.opt.option_type == OptionType.PUT:
            payoff = np.maximum(k - price_matrix[:, -1], 0)
        else:
            raise ValueError("Option type incorrect")
        return payoff







class PayoffFactory:
    @staticmethod
    def create(opt: Option) -> PayoffCalculator:
        if isinstance(opt, BarrierOption):
            return BarrierPayoffCalculator(opt)
        if isinstance(opt, Option):
            return VanillaPayoffCalculator(opt)

        raise ValueError("Produit non supporté")