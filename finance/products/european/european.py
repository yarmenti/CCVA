from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from maths.montecarlo.processes.base import Process
from finance.discountfactor import DiscountFactor
from enum import Enum


class EuropeanContract(object):
    __metaclass__ = ABCMeta

    def __init__(self, underlying_path, maturity, df_process, underlying_index):
        #if not isinstance(underlying_path, Process):
        #    raise ValueError("The underlying must be of type Path")

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

    @abstractproperty
    def asset_class(self):
        pass

    @abstractmethod
    def coupon(self, t):
        return 0.

    @abstractmethod
    def price(self, t):
        pass

    @property
    def pillars(self):
        return np.array([0., self.maturity], copy=True)

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


class ContractType(Enum):
    interest_rate = 1
    curr_rates = 2
    gold = 3
    equity = 4
    precious_metal = 5
    ig_cds = 6
    nig_cds = 7
    other = 8
