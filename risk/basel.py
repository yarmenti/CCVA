# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:32:58 2015

@author: Yann
"""

import numpy as np
from scipy.stats import norm

class RegulatoryCapital(object):
    __systemic_fator__ = norm.ppf(0.999)
    
    def __init__(self, vm_accounts, im_accounts, portfolio, **kwargs):
        self._vm_ = vm_accounts
        self._im_ = im_accounts
        
        self._port_ = portfolio
        
        self._init_kwargs_(**kwargs)
        
    def _init_kwargs_(self, **kwargs):
        self._recovery_ = kwargs.get('recovery', None)
        self._dp_model_= kwargs.get('default_proba', None)
        self._c_= kwargs.get('c', 1.)
        
    @classmethod
    def __b__(cls, x):
        return (0.11852 - 0.05478 * np.log(x))**2        
        
    @classmethod
    def __compute_correl__(cls, default_proba):
        one_minus_exp_m50 = 1. - np.exp(-50)
        one_minus_exp_m50_pd = 1. - np.exp(-50*default_proba)
        
        term1 = 0.12* one_minus_exp_m50_pd / one_minus_exp_m50
        term2 = 0.24*(1.- one_minus_exp_m50_pd) / one_minus_exp_m50

        return term1+term2
        
    def __compute_effective_mat__(self, index, t):
        den = self._port_.amounts[index]
        
        ttm = [d.maturity-t for d in self._port_.derivatives]        
        notionals = self._port_.weights[index, :] * den
        num = np.dot(ttm, notionals)
                
        maxx = np.maximum(1., num/den)
        res = np.minimum(5., maxx)
        
        return res
        
    def _compute_regulatory_capital_(self, index, t, **kwargs):
        recov = kwargs.get('recovery', self._recovery_)
        lgd = 1.-recov
        
        default_model = kwargs.get('default_proba', self._dp_model_)
        exposure_mat = kwargs.get('exposure_mat', 1.)
        dp = default_model.default_proba(exposure_mat)
        
        correl = self.__compute_correl__(dp)
        
        tmp = norm.ppf(dp)+np.sqrt(correl)*self.__systemic_fator__
        gauss_factor = norm.cdf(tmp / np.sqrt(1.-correl))
        
        M = self.__compute_effective_mat__(index, t)
        b_dp = self.__b__(dp)
        
        coeff = (1.+(M-2.5)*b_dp)/(1.-1.5*b_dp)
                
        return lgd*(gauss_factor-dp)*coeff        
        
    def _compute_ead_(self, index, t, **kwargs):
        ebrms = self._port_.compute_exposure(t)

        vms = self._vm_.amounts
        ims = self._im_.amounts        
        
        ead = ebrms[index, 0] - vms[index] - ims[index]
        ead = np.maximum(ead, 0.)
        
        return ead
        
    def compute_rwa(self, index, t, **kwargs):
        K = self._compute_regulatory_capital_(index, t, **kwargs)
        ead = self._compute_ead_(index, t, **kwargs)
        
        return 12.5*ead*K*self._c_