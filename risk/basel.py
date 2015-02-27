# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:32:58 2015

@author: Yann
"""

import numpy as np
from scipy.stats import norm

import abc

class BaselRegulatoryCapital(object):
    __metaclass__ = abc.ABCMeta
    __systemic_fator__ = norm.ppf(0.999)    
    
    @classmethod
    def __b__(cls, x):
        return (0.11852 - 0.05478 * np.log(x))**2
        
    @classmethod
    def __compute_correl__(cls, default_proba):
        return 0.12+0.12*(1+np.exp(-50*default_proba))
        
    @abc.abstractmethod
    def compute_regulatory_capital(self, **kwargs):
        pass

    def compute_rwa(self, ead, **kwargs):
        K = self.compute_regulatory_capital(**kwargs)
        return 12.5*ead*K
    
class BaselSwapRegulatoryCapital(BaselRegulatoryCapital):
    def __init__(self, swap, **kwargs):
        self.__swap__ = swap
        self._maturity_ = swap.maturity
        self._init_constants_(**kwargs)
            
    def _init_constants_(self, **kwargs):
        self._recovery_ = kwargs.get('recovery', None)
        self._dp_model_= kwargs.get('default_proba', None)
        
    def compute_regulatory_capital(self, **kwargs):
        recov = kwargs.get('recovery', self._recovery_)
        lgd = 1.-recov
        
        default_model = kwargs.get('default_proba', self._dp_model_)
        exposure_mat = kwargs.get('exposure_mat', 1.)
        dp = default_model.default_proba(exposure_mat)
        
        correl = self.__compute_correl__(dp)
        
        tmp = norm.ppf(dp)+np.sqrt(correl)*self.__systemic_fator__
        gauss_factor = norm.cdf(tmp / np.sqrt(1.-correl))
        
        M = kwargs.get('maturity', self._maturity_)
        b_dp = self.__b__(dp)
        coeff = (1.+(M-2.5)*b_dp)/(1.-1.5*b_dp)
                
        return lgd*(gauss_factor-dp)*coeff
    
BaselRegulatoryCapital.register(BaselSwapRegulatoryCapital)