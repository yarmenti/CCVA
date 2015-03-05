# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:38:07 2015

@author: Yann
"""

from abc import ABCMeta, abstractmethod
import warnings

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
        #derivative_index = kwargs.get('derivative_index')        
        derivative_index = list(portfolio.derivatives).index(self._contract_)
        
        weights = portfolio.weights[:, derivative_index]

        risk_period = kwargs.get('risk_period', self._risk_period_)        
        conf_level = kwargs.get('conf_level', self._conf_level_)
        df = kwargs.get('conf_level', self._df_)

        if t+risk_period > self._contract_.maturity:
            risk_period = self._contract_.maturity - t

        res = self._v_func_(t, risk_period, self._drift_, self._vol_, conf_level, weights, df)
   
        return res

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