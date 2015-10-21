import numpy as np

from maths.montecarlo.processes.base import Process


class HistoricalProcess(Process):
    def __init__(self, time, values):
        if isinstance(values, list):
            values = np.array(values)

        x0 = None
        if values.ndim == 1:
            x0 = [values[0]]
        elif values.ndim == 2:
            x0 = values[:, 0]
        else:
            raise ValueError("The values must be of dimension 2 or 1, given: %s"%values.ndim)

        super(HistoricalProcess, self).__init__(time, x0)
        self.values = values

        def func():
            raise NotImplementedError("Cannot change time for historical process")

        self._time_set = func

    def _time_set(self):
        pass

    def conditional_expectation(self, t, T):
        return self(t)

    def simulate(self):
        raise NotImplementedError("Cannot simulate historical process")

Process.register(HistoricalProcess)