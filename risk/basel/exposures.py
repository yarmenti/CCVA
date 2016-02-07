import numpy as np
from finance.products.european.european import ContractType


class SABaselExposure(object):
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


class BaselExposureAtDefault(object):
    def __init__(self, portfolio, eee_func_array):
        self.__port = portfolio
        self.__eee = eee_func_array

    @property
    def portfolio(self):
        return self.__port
    
    def __call__(self, **kwargs):
        t = kwargs['t']
        epsilon = kwargs.pop('epsilon')
        positions = kwargs.pop('positions', self.__port.positions)
        ttm = kwargs.pop("ttm", 1.)

        notionals = self.__port.notionals

        res = np.empty(positions.shape)
        for i, (d, eee) in enumerate(zip(self.__port.derivatives, self.__eee)):
            mat = d.maturity
            time_discr = np.arange(t + epsilon,
                                   np.minimum( t + ttm + epsilon, mat) + epsilon, 
                                   epsilon)

            if len(time_discr) == 1:
                time_discr = np.arange(t + 0.5*epsilon,
                                   np.minimum( t + ttm + 0.5*epsilon, mat) + 0.5*epsilon, 
                                   0.5*epsilon)

            previous = np.zeros(positions.shape[0])
            sum = np.zeros(positions.shape[0])
            pos = positions[:, i].reshape((len(positions[:, i]), 1))
            for t_i in time_discr:
                current = eee(t_i, pos, **kwargs).flatten()
                current = np.maximum(previous, current)

                sum += current
                previous = current

            res[:, i] = 1.4 * epsilon * sum * notionals[i]

        return res
