from abc import ABCMeta, abstractmethod
from scipy.stats import norm
from finance.products.european.european import ContractType

from maths.montecarlo.processes.brownianmotions import BrownianMotion, GeometricBrownianMotion
from maths.montecarlo.processes.historical import HistoricalProcess
from utils import time_offseter

import numpy as np


class BaselExposure(object):
    __metaclass__ = ABCMeta

    def __init__(self, european_contract):
        self.__contract = european_contract

    @property
    def contract(self):
        return self.__contract

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class AbsEuropeanBaselExposure(BaselExposure):
    def __call__(self, **kwargs):
        t = kwargs.pop('t')

        risk_horizon = kwargs['risk_horizon']
        if t+risk_horizon > self.contract.maturity:
            risk_horizon = self.contract.maturity - t

        risk_period = kwargs['risk_period']

        alpha = kwargs['conf_level']
        conf_level = [alpha, 1.-alpha]

        self.processes = []

        time_grid = self.contract.underlying.time
        if risk_horizon < 0:
            tph = time_offseter(t+risk_period, time_grid, time_grid[-1])
            keep_pills = self.contract.pillars[t <= self.contract.pillars]
            keep_pills = keep_pills[keep_pills <= tph]
            time_grid = np.append(keep_pills, [t, tph])
            time_grid.sort()
        else:
            time_grid = time_grid[(t <= time_grid) & (time_grid <= t+risk_horizon)]

        time_grid = np.unique(time_grid)
        old_udl = self.contract.underlying

        res = np.zeros([2])
        for i, q in enumerate(conf_level):
            quantile_inv = norm.ppf(q)
            process = self._simulate_process(t, time_grid, quantile_inv, **kwargs)

            self.contract.underlying = process

            time_grid_ = process.time[process.time >= t]
            for t_ in time_grid_:
                t_ph = time_offseter(t_+risk_period, time_grid, time_grid[-1])
                tmp = self.contract.p_and_l(t_, t_ph)
                if abs(tmp) >= abs(res[i]):
                    res[i] = tmp

            self.processes.append(process)

        self.contract.underlying = old_udl
        return res

    @abstractmethod
    def _simulate_process(self, t, time_grid, quantile_inv, **kwargs):
        pass


class AbsUdlEuropeanBaselExposure(AbsEuropeanBaselExposure):
    def __init__(self, european_contract, drift=None, vol=None):
        super(AbsUdlEuropeanBaselExposure, self).__init__(european_contract)
        self.__drift = drift
        self.__vol = vol

    def _simulate_process(self, t, time_grid, quantile_inv, **kwargs):
        drift = kwargs.get("drift", self.__drift)
        vol = kwargs.get("vol", self.__vol)

        if drift is None or vol is None:
            raise ValueError("Drift or vol can't be None")

        sim_times = time_grid[1:]
        sim_values = []

        sim_values.append(self.contract.S(t))
        t_ = t

        for t__ in sim_times:
            current = sim_values[-1]
            future = self.compute_next(current, drift, vol, t__-t_, quantile_inv)[0, 0]
            sim_values.append(future)

            t_ = t__

        sim_values = sim_values[1:]

        historical_times = self.contract.underlying.time
        historical_times = historical_times[historical_times <= t].tolist()
        historical_values = [self.contract.S(t__) for t__ in historical_times]

        process_time = historical_times + sim_times.tolist()
        process_vals = historical_values + sim_values

        return HistoricalProcess(process_time, process_vals)

    @abstractmethod
    def compute_next(self, current, drift, vol, delta_t, quantile_inv):
        pass


class EuropeanBaselBrownianExposure(AbsUdlEuropeanBaselExposure):
    def compute_next(self, current, drift, vol, delta_t, quantile_inv):
        return BrownianMotion.compute_next_value(current, drift, vol, delta_t, quantile_inv)


class EuropeanBaselGeometricBrownianExposure(AbsUdlEuropeanBaselExposure):
    def compute_next(self, current, drift, vol, delta_t, quantile_inv):
        return GeometricBrownianMotion.compute_next_value(current, drift, vol, delta_t, quantile_inv)


class SABaselExposure(BaselExposure):
    __low = 0
    __middle = 1
    __high = 2

    __mapping = {ContractType.interest_rate: [0., 0.005, 0.015],
                 ContractType.curr_rates: [0.01, 0.05, 0.075],
                 ContractType.gold: [0.01, 0.05, 0.075],
                 ContractType.equity: [0.06, 0.08, 0.1],
                 ContractType.precious_metal: [0.07, 0.07, 0.08],
                 ContractType.ig_cds: [0.05, 0.05, 0.05],
                 ContractType.nig_cds: [0.1, 0.1, 0.1],
                 ContractType.other: [0.10, 0.12, 0.15]}

    def __init__(self, portfolio, mult=0.15):
        self.__port = portfolio
        self.__mult = mult

    @classmethod
    def __residual_maturity(cls, time_to_maturity):
        if time_to_maturity < 1.:
            return cls.__low
        elif 1. <= time_to_maturity < 5.:
            return cls.__middle
        else:
            return cls.__high

    def __call__(self, **kwargs):
        t = kwargs.pop('t')

        res = np.zeros(self.__port.positions.shape)
        for i, d in enumerate(self.__port.derivatives):
            ttm = d.maturity - t
            index = self.__residual_maturity(ttm)
            mult_coeff = self.__mapping[d.asset_class][index] * self.__mult

            res[:, i] = mult_coeff * self.__port.notionals[:, i]

        return res

AbsUdlEuropeanBaselExposure.register(EuropeanBaselBrownianExposure)
AbsUdlEuropeanBaselExposure.register(EuropeanBaselGeometricBrownianExposure)
BaselExposure.register(SABaselExposure)