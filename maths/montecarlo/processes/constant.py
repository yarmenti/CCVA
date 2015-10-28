import numpy as np

from maths.montecarlo.processes.base import Process


class ConstantProcess(Process):
    def __init__(self, time, x_0):
        super(ConstantProcess, self).__init__(time, x_0)
        self.simulate()

    def _time_set(self):
        self.simulate()

    def conditional_expectation(self, T, t):
        return self(t)

    def simulate(self):
        values = np.tile(self._x0, self.time.shape)
        super(ConstantProcess, self.__class__).values.fset(self, values)

Process.register(ConstantProcess)