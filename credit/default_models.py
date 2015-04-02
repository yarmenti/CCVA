# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:42:15 2015

@author: Yann
"""

import abc
import numpy as np

class DefaultableModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, recovery):
        if recovery < 0 or recovery > 1:
            raise ValueError("The recovery must lie in [0, 1]")

        self.__r = recovery

    @property
    def recovery(self):
        return self.__r

    @abc.abstractmethod
    def survival_proba(self, t):
        pass

    def default_proba(self, t):
        return 1. - self.survival_proba(t)

class FlatIntensity(DefaultableModel):
    def __init__(self, intensity, recovery):
        super(FlatIntensity, self).__init__(recovery)
        if intensity < 0:
            raise ValueError("The intensity must be non negative")

        self.__lambda = intensity

    def survival_proba(self, t):
        if t == 0.:
            return 1.

        return np.exp(-self.__lambda*t)

class StepwiseConstantIntensity(DefaultableModel):
    def __init__(self, pillars, hazard_rates, recovery):
        super(StepwiseConstantIntensity, self).__init__(recovery)

        if np.min(hazard_rates) < 0:
            raise ValueError("The hazard rates must be non negative")

        self.__pill = np.array(pillars)
        self.__hazard_rates = np.cumsum(hazard_rates)
        self.__hzrd_max_index = len(self.__hazard_rates)-1

        if self.__pill.shape != self.__hazard_rates.shape:
            raise ValueError("The length of the pillars and hazard_rates must be the same.")

        self.__pill = np.insert(self.__pill, 0, 0)

        self.__init_log_probas()

    @property
    def pillars(self):
        return np.array(self.__pill, copy=True)

    @property
    def intensities(self):
        return np.array(self.__hazard_rates, copy=True)

    def __init_log_probas(self):
        deltas = np.ediff1d(self.__pill)
        tmp = np.multiply(self.__hazard_rates, deltas)
        self.__log_probs = np.insert(np.cumsum(tmp), 0, 0.)

    def survival_proba(self, t):
        if t == 0.:
            return 1.

        index = np.searchsorted(self.__pill, t, side='left')-1
        cum_sum = self.__log_probs[index]
        missing = (t - self.__pill[index])*self.__hazard_rates[np.minimum(index, self.__hzrd_max_index)]

        return np.exp(-missing - cum_sum)

DefaultableModel.register(FlatIntensity)
DefaultableModel.register(StepwiseConstantIntensity)