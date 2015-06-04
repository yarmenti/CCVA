from __builtin__ import property
import numpy as np


class ConstantSkinInTheGame(object):
    def __init__(self, initial_value):
        assert (initial_value >= 0), "The initial value is not positive"
        self.__init_val = initial_value
        self.__val = initial_value

    def update_value(self, t, **kwargs):
        pass

    def handle_breach(self, breach):
        jump = -np.minimum(breach, self.__val)
        self.__val += jump
        self.__val = np.maximum(self.__val, 0)
        return breach+jump

    @property
    def value(self):
        return self.__val
    
    def recover(self, val=None):
        self.__val = self.__init_val if val is None else val


class SkinInTheGame(object):
    def __init__(self, initial_value=0., ratio=0.25):
        if initial_value < 0:
            raise ValueError("The initial_value must be non negative.")

        self.__val = initial_value
        self.__init_val = initial_value

        self.__ratio = ratio

    def update_value(self, t, **kwargs):
        regul_capital = kwargs.pop("regul_capital")
        risk_horizon = kwargs.pop("risk_horizon", -1)

        if "losses" not in kwargs:
            raise ValueError("The losses must be present if the risk horizon or conf_level are not.")

        self.__val = regul_capital.compute_k_ccp(t, risk_horizon, **kwargs) * self.__ratio

    def handle_breach(self, breach):
        jump = -np.minimum(breach, self.__val)
        self.__val += jump
        self.__val = np.maximum(self.__val, 0)
        return breach + jump

    def recover(self, val=None):
        self.__val = self.__init_val if val is None else val

    @property
    def value(self):
        return self.__val