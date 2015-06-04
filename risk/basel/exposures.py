from abc import ABCMeta, abstractmethod
from finance.products.european.european import ContractType

import numpy as np


class BaselExposure(object):
    __metaclass__ = ABCMeta

    def __init__(self, european_contract):
        self.__contract = european_contract

    @property
    def contract(self):
        return self.__contract

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class SABaselExposure(BaselExposure):
    __low = 0
    __middle = 1
    __high = 2

    __mapping = {ContractType.interest_rate: [0., 0.005, 0.015],
                 ContractType.curr_rates: [0.01, 0.05, 0.075],
                 ContractType.gold: [0.01, 0.05, 0.075],
                 ContractType.equity: [0.06, 0.08, 0.1],
                 ContractType.precious_metal: [0.07, 0.07, 0.08],
                 ContractType.ig_cds: [0.05, 0.05, 0.05],
                 ContractType.nig_cds: [0.1, 0.1, 0.1],
                 ContractType.other: [0.10, 0.12, 0.15]}

    def __init__(self, portfolio, mult=0.15):
        self.__port = portfolio
        self.__mult = mult

    @classmethod
    def __residual_maturity(cls, time_to_maturity):
        if time_to_maturity < 1.:
            return cls.__low
        elif 1. <= time_to_maturity < 5.:
            return cls.__middle
        else:
            return cls.__high

    def __call__(self, **kwargs):
        t = kwargs.pop('t')

        res = np.zeros(self.__port.positions.shape)
        for i, d in enumerate(self.__port.derivatives):
            ttm = d.maturity - t
            index = self.__residual_maturity(ttm)
            mult_coeff = self.__mapping[d.asset_class][index] * self.__mult

            res[:, i] = mult_coeff * self.__port.notionals[:, i]

        return res

BaselExposure.register(SABaselExposure)