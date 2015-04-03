# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:38:07 2015

@author: Yann
"""

from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from maths.montecarlo.deterministicpath import DeterministicPath
import numpy as np


class Exposure(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, risk_period):
        self._risk_period_ = risk_period

    @abstractmethod
    def __call__(self, **kwargs):
        pass
    
#####################################################################    


class MocExposure(Exposure):
    def __init__(self):
        pass
    
    def __call__(self, **kwargs):
        raise NotImplementedError()

#####################################################################


class EuropeanQuantileBrownianExposure(Exposure):
    def __init__(self, european_contract, df, drift, vol):
        self.__vol = vol
        self.__drift = drift
        self.__df = df
        self.__contract = european_contract

    def __call__(self, **kwargs):
        t = kwargs.get('t')
        risk_period = kwargs.get('risk_period')
        if t+risk_period > self.__contract.maturity:
            risk_period = self.__contract.maturity - t

        alpha = kwargs.get('conf_level')
        conf_level = [alpha, 1.-alpha]

        drift = kwargs.get('drift', self.__drift)
        vol = kwargs.get('vol', self.__vol)

        res = np.zeros([2])
        for (i_, a) in enumerate(conf_level):
            tmp = self.__compute_pl_price(t, risk_period, a, drift, vol)
            res[i_] = tmp

        return res

    def __compute_pl_price(self, t, risk_period, alpha, drift, vol):
        S_t = self.__contract.S(t)
        t_ph = t+risk_period
        quantile_inv = norm.ppf(alpha)

        lst_pill_idx = np.searchsorted(self.__contract.pillars, t_ph)
        pillars = np.unique(np.append(self.__contract.pillars[:lst_pill_idx], t_ph))
        process_values = {}
        for t_ in pillars:
            if t_ <= t:
                St_ = self.__contract.S(t_)
            else:
                St_ = St_ + drift*(t_-t__) + vol*np.sqrt(t_-t__)*quantile_inv

            process_values[t_] = St_
            t__ = t_
            St__ = St_

        def f(x):
            res = np.empty(self.__contract.underlying_index + 1)
            res.fill(process_values[x])
            return res.tolist()

        tmp_underlying = self.__contract.underlying
        self.__contract.underlying = DeterministicPath(f, process_values.keys())

        result = self.__contract.price(t_ph) - self.__contract.price(t)

        self.__contract.underlying = tmp_underlying

        return result

#####################################################################

Exposure.register(MocExposure)
Exposure.register(EuropeanQuantileBrownianExposure)