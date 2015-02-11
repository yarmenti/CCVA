# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:37:27 2015

@author: Yann
"""

from .european import EuropeanContract

class FutureContract(EuropeanContract):
    def __init__(self, underlying, df_process, maturity, underlying_index=0):
        super(FutureContract, self).__init__(underlying, maturity, df_process, underlying_index)        
        self._T_ = maturity        
        self._df_mat_ = self._df_(self._T_)
        
    def price(self, t):
        if t > self._T_:
            return 0

        S_t = self._get_St_(t)
        res = self._df_(t)/self._df_mat_ * S_t
        return res
    
    def __str__(self):
        return "Future contract of maturity T = %d years over S^%d"%(self.maturity, self._underlying_index_)    
    
EuropeanContract.register(FutureContract)

#####################################################################

from scipy.stats import norm
import numpy as np

def ind_future_brownian_quantile(t, h, drift_t, vol_t, alpha, future, weight, df):
    if t > future.maturity:
        return 0.
        
    if t+h > future.maturity:
        hp = future.maturity - t
        return ind_future_brownian_quantile(t, hp, drift_t, vol_t, alpha, future, weight, df)
    
    S_t = future.underlying(t)[future.underlying_index]
    F_t = future.price(t)
    
    discount = df(t+h)/df(future.maturity)

    conf_level = alpha if weight>0 else 1-alpha
    quantile_inv = norm.ppf(conf_level)
    
    S_t_h = S_t + (drift_t*h) + (vol_t*np.sqrt(h) * quantile_inv)
    
    return weight * (discount * S_t_h - F_t)