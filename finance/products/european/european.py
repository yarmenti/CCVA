# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:30:02 2015

@author: Yann
"""

from abc import ABCMeta, abstractmethod
from maths.montecarlo.path import Path, Process
from finance.discountfactor import DiscountFactor

import numpy as np
from scipy.stats import norm        
from maths.montecarlo.deterministicpath import DeterministicPath


class EuropeanContract(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, underlying_path, maturity, df_process, underlying_index):
        if not isinstance(underlying_path, (Path, Process)):
            raise ValueError("The underlying must be of type Path")

        if maturity not in underlying_path.time:
            raise ValueError("The Maturity is not in the time grid")
        
        self.__mat = maturity
        self.__udlyg = underlying_path
        self.__udlyg_idx = underlying_index

        if not isinstance(df_process, DiscountFactor):
            raise ValueError("The df_process must be of type DiscountFactor")

        self.__df = df_process

    def S(self, t):
        return self.__udlyg(t)[self.__udlyg_idx, 0]

    def p_and_l(self, t1, t2):
        if t1 > t2:
            raise ValueError("t1 must be lower than t2")
    
        if t1 < self.__mat < t2:
            t2 = self.__mat
        
        return self.price(t2) - self.price(t1)     
    
    @property
    def pillars(self):
        return np.array([0., self.maturity], copy=True)
    
    @abstractmethod
    def price(self, t):
        pass
    
    @property
    def maturity(self):
        return self.__mat
    
    @property
    def discount_factor(self):
        return self.__df

    @property
    def underlying(self):
        return self.__udlyg

    @underlying.setter
    def underlying(self, value):
        self.__udlyg = value

    @property
    def underlying_index(self):
        return self.__udlyg_idx
        
    def __additional_points_subprocess__(self, **kwargs):
        return dict()