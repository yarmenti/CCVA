import numpy as np
from abc import ABCMeta, abstractmethod


class DiscountFactor(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(time):
        pass


class ConstantRateDiscountFactor(DiscountFactor):
    def __init__(self, rate):
        self.__r = rate
        
    def __call__(self, t):
        return np.exp(-self.__r*t)

    @property
    def rate(self):
        return self.__r

    def __str__(self):
        return "Constant discount factor process with rate r = %.2f"%self.__r
        
DiscountFactor.register(ConstantRateDiscountFactor)