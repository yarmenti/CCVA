# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:38:07 2015

@author: Yann
"""

from scipy.stats import norm
from abc import ABCMeta, abstractmethod
import warnings
from maths.montecarlo.deterministicpath import DeterministicPath


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
    def __init__(self, european_contract, risk_period, drift, vol, conf_level, df):
        super(EuropeanQuantileBrownianExposure, self).__init__(risk_period)

        self._contract_ = european_contract
        
        self._drift_ = drift
        self._vol_ = vol
        self._conf_level_ = conf_level
        self._df_ = df
        
        self._v_func_ = np.vectorize(self._contract_.compute_brownian_quantile_price, excluded=['t', 'h', 'drift_t', 'vol_t', 'alpha', 'df'])
        
    def __call__(self, **kwargs):
        t = kwargs.get('t')
        portfolio = kwargs.get('portfolio')
        derivative_index = list(portfolio.derivatives).index(self._contract_)
        
        weights = portfolio.weights[:, derivative_index]
        risk_period = kwargs.get('risk_period', self._risk_period_)        
        conf_level = kwargs.get('conf_level', self._conf_level_)
        
        df = kwargs.get('conf_level', self._df_)

        if t+risk_period > self._contract_.maturity:
            risk_period = self._contract_.maturity - t

        res = self._v_func_(t, risk_period, self._drift_, self._vol_, conf_level, weights, df)
        
        return res.reshape(res.size , 1)

class EuropeanQuantileBrownianExposureV2(Exposure):
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
        S_t = self.__contract._get_St_(t)
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
        self.__contract.set_underlying(DeterministicPath(f, process_values.keys()))

        result = self.__contract.price(t_ph) - self.__contract.price(t)

        self.__contract.set_underlying(tmp_underlying)

        return result

#####################################################################

import numpy as np
from finance.products.european.forward import ind_forward_brownian_quantile  
    
class ForwardQuantileExposure(Exposure):
    def __init__(self, risk_period, drift, vol, conf_level, df):
        warnings.warn("Call to deprecated class", 9, category=DeprecationWarning)
        super(ForwardQuantileExposure, self).__init__(risk_period)
        
        self._drift_ = drift
        self._vol_ = vol
        self._conf_level_ = conf_level
        self._df_ = df
        
        self._v_func_ = np.vectorize(ind_forward_brownian_quantile, excluded=['t', 'h', 'drift_t', 'vol_t', 'alpha', 'forward', 'df'])        
        
    def __call__(self, t, portfolio, derivative_index, **kwargs):
        forward = portfolio.derivatives[derivative_index]
        weights = portfolio.weights[:, derivative_index]

        risk_period = kwargs.get('risk_period', self._risk_period_)        
        conf_level = kwargs.get('conf_level', self._conf_level_)   
        
        res = self._v_func_(t, risk_period, self._drift_, self._vol_, conf_level, forward, weights, self._df_)
        
        return res
        
Exposure.register(ForwardQuantileExposure)

#####################################################################    

from finance.products.european.future import ind_future_brownian_quantile  

class FutureQuantileExposure(Exposure):
    def __init__(self, risk_period, drift, vol, conf_level, df):        
        warnings.warn("Call to deprecated class", 9, category=DeprecationWarning)
        super(FutureQuantileExposure, self).__init__(risk_period)
        
        self._drift_ = drift
        self._vol_ = vol
        self._conf_level_ = conf_level
        self._df_ = df
        
        self._v_func_ = np.vectorize(ind_future_brownian_quantile, excluded=['t', 'h', 'drift_t', 'vol_t', 'alpha', 'future', 'df'])
        
    def __call__(self, t, portfolio, derivative_index, **kwargs):
        future = portfolio.derivatives[derivative_index]
        weights = portfolio.weights[:, derivative_index]
        
        risk_period = kwargs.get('risk_period', self._risk_period_)        
        conf_level = kwargs.get('conf_level', self._conf_level_)
        
        res = self._v_func_(t, risk_period, self._drift_, self._vol_, conf_level, future, weights, self._df_)
        
        return res
        
Exposure.register(FutureQuantileExposure)