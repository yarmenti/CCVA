from abc import ABCMeta, abstractmethod
from scipy.stats import norm

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
        if t+risk_horizon> self.contract.maturity:
            risk_horizon = self.contract.maturity - t

        risk_period = kwargs['risk_period']

        alpha = kwargs['conf_level']
        conf_level = [alpha, 1.-alpha]

        #conf_level = kwargs['conf_level']
        #quantile_inv = norm.ppf(conf_level)

        #time_grid = self.contract.underlying.time
        #time_grid = time_grid[(t <= time_grid) & (time_grid <= t+risk_horizon)]

        # process = self._simulate_process(t, time_grid, quantile_inv, **kwargs)
        # old_udl = self.contract.underlying
        # self.contract.underlying = process
        #
        # time_grid = process.time[process.time >= t]
        #
        # res = 0.
        # for t_ in time_grid:
        #     t_ph = t_ + risk_period
        #     t_ph = time_offseter(t_ph, time_grid, time_grid[-1])
        #
        #     tmp = self.contract.p_and_l(t_, t_ph)
        #
        #     if tmp >= res:
        #         res = tmp
        #
        # self.contract.underlying = old_udl
        #
        # self.process = process

        self.processes = []

        time_grid = self.contract.underlying.time
        time_grid = time_grid[(t <= time_grid) & (time_grid <= t+risk_horizon)]

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

                if tmp >= res[i]:
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



AbsUdlEuropeanBaselExposure.register(EuropeanBaselBrownianExposure)
AbsUdlEuropeanBaselExposure.register(EuropeanBaselGeometricBrownianExposure)