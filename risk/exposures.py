from abc import ABCMeta, abstractmethod

from scipy.stats import norm
import numpy as np

from maths.montecarlo.processes.historical import HistoricalProcess


class Exposure(object):
    __metaclass__ = ABCMeta

    def __init__(self, risk_period):
        self._risk_period_ = risk_period

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class EuropeanQuantilExposure(Exposure):
    def __init__(self, european_contract, df, drift, vol):
        if isinstance(vol, (list, np.ndarray)):
            if len(vol) != 1:
                raise ValueError("Vol must be a scalar")
            vol = vol[0]
        self.__vol = vol

        if isinstance(drift, (list, np.ndarray)):
            if len(drift) != 1:
                raise ValueError("Drift must be a scalar")
            drift = drift[0]
        self.__drift = drift

        self.__df = df
        self.__contract = european_contract

    def __call__(self, **kwargs):
        t = kwargs['t']
        risk_period = kwargs['risk_period']
        if t+risk_period > self.__contract.maturity:
            risk_period = self.__contract.maturity - t

        alpha = kwargs['conf_level']
        conf_level = [alpha, 1.-alpha]

        drift = kwargs.get('drift', self.__drift)
        vol = kwargs.get('vol', self.__vol)

        res = np.zeros([2])
        for (i_, a) in enumerate(conf_level):
            tmp = self.__compute_pl_price(t, risk_period, a, drift, vol)
            res[i_] = tmp

        return res

    def __compute_pl_price(self, t, risk_period, alpha, drift, vol):
        t_ph = t+risk_period
        quantile_inv = norm.ppf(alpha)

        lst_pill_idx = np.searchsorted(self.__contract.pillars, t_ph)
        pillars = np.unique(np.append(self.__contract.pillars[:lst_pill_idx], [t, t_ph]))
        pillars.sort()

        process_values = self._compute_diff(t, pillars, drift, vol, quantile_inv)

        tmp_underlying = self.__contract.underlying
        values = np.tile(process_values, (self.__contract.underlying.dimension, 1))
        self.__contract.underlying = HistoricalProcess(pillars, values)

        result = self.__contract.p_and_l(t, t_ph)

        self.__contract.underlying = tmp_underlying

        return result

    @abstractmethod
    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        pass

    @property
    def contract(self):
        return self.__contract


class EuropeanQuantileBrownianExposure(EuropeanQuantilExposure):

    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        process_values = []
        for t_ in pillars:
            if t_ <= t:
                St_ = self.contract.S(t_)
            else:
                St_ = St_ + drift*(t_-t__) + vol*np.sqrt(t_-t__)*quantile_inv

            process_values.append(St_)
            t__ = t_

        return process_values


class EuropeanQuantileGeomBrownianExposure(EuropeanQuantilExposure):
    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        d = drift - .5*vol**2

        process_values = []
        for t_ in pillars:
            if t_ <= t:
                St_ = self.contract.S(t_)
            else:
                St_ = St_*np.exp(d*(t_-t__) + vol*np.sqrt(t_-t__)*quantile_inv)

            process_values.append(St_)
            t__ = t_

        return process_values

EuropeanQuantilExposure.register(EuropeanQuantileBrownianExposure)
EuropeanQuantilExposure.register(EuropeanQuantileGeomBrownianExposure)