# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:30:02 2015

@author: Yann
"""

from abc import ABCMeta, abstractmethod
from maths.montecarlo.path import Path
from finance.discountfactor import DiscountFactor

import numpy as np
from scipy.stats import norm        
from maths.montecarlo.deterministicpath import DeterministicPath

class EuropeanContract(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, underlying_path, maturity, df_process, underlying_index):
        assert isinstance(underlying_path, Path), "The underlying must be of type Path"        
        assert maturity in underlying_path.time, "The Maturity is not in the time grid" 
        
        self._T_ = maturity
        self._underlying_ = underlying_path
        self._underlying_index_ = underlying_index        
        
        assert isinstance(df_process, DiscountFactor), "The df_process must be of type DiscountFactor"
        self._df_ = df_process
    
    def _get_St_(self, t):
        return self._underlying_(t)[self._underlying_index_][0, 0]

    def S(self, t):
        return self._get_St_(t)

    def p_and_l(self, t1, t2):
        assert (t1<=t2), "t1 must be lower than t2"
    
        if t1 < self._T_ and self._T_ < t2:
            t2 = self._T_
        
        return self.price(t2) - self.price(t1)     
    
    @property
    def pillars(self):
        return np.array([0., self.maturity])
    
    @abstractmethod
    def price(self, t):
        pass
    
    @property
    def maturity(self):
        return self._T_
    
    @property
    def discount_factor(self):
        return self._df_
    
    def set_underlying(self, underlying):
        self._underlying_ = underlying
    
    @property
    def underlying(self):
        return self._underlying_
        
    @property
    def underlying_index(self):
        return self._underlying_index_
        
    def __additional_points_subprocess__(self, **kwargs):
        return dict()
        
    def compute_brownian_quantile_price(self, t, h, drift_t, vol_t, alpha=0.95, weight=1., df=None):
        if df is None:
            df = self.discount_factor
        
        S_t = self._get_St_(t)
        
        t_ph = t+h
        conf_level = alpha if weight>0 else 1-alpha
        quantile_inv = norm.ppf(conf_level)
        
        S_th = S_t + drift_t*h + vol_t*np.sqrt(h)*quantile_inv
        
        process_values = {0: 0., t: S_t, t_ph: S_th}
        process_values.update(self.__additional_points_subprocess__(
            t=t, 
            t_ph=t_ph,
            current=process_values
        ))
        
        def f(x):
            res = np.empty(self._underlying_index_ + 1)
            res.fill(process_values[x])
            return res.tolist()

        tmp_underlying = self.underlying
        self.set_underlying(DeterministicPath(f, process_values.keys()))

        res = self.price(t_ph) - self.price(t)
        res *= weight        
        
        self.set_underlying(tmp_underlying)
        
        return np.maximum(res, 0.)