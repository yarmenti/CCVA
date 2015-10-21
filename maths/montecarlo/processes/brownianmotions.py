import numpy as np

from maths.montecarlo.processes.base import Process


class BrownianMotion(Process):
    __drifts = None
    __vols = None

    def __init__(self, time, w0, drifts, vols, correl_matrix=None):
        super(BrownianMotion, self).__init__(time, w0)

        if self.dimension != 1 and correl_matrix is None:
            raise ValueError("The correl_matrix can't be None when dimension = %s"%self.dimension)

        self.correl_matrix = correl_matrix
        self.__drifts = self.__check_drifts_and_vols_init(drifts)
        self.__vols = self.__check_drifts_and_vols_init(vols)
        self.simulate()

    def __check_drifts_and_vols_init(self, vector):
        if isinstance(vector, (float, int)):
            tmp = np.zeros((self.dimension, 1))
            tmp.fill(vector)
            vector = tmp
            return vector

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector).reshape(self.dimension, 1)

        if vector.ndim != 2:
            raise ValueError("The number of dimension must be 2")

        # We can't put a matrix because of
        # setting time

        #dim_, time_ = vector.shape
        #if dim_ == self.dimension and time_ == 1:
        #    vector = np.tile(vector, self.time.size-1)

        return vector

    def _time_set(self):
        self._delta_time = np.tile(np.ediff1d(self.time), (self.dimension, 1))
        self._sqrt_delta_time = np.sqrt(self._delta_time)

        if self.__vols is not None and self.__drifts is not None:
            self.simulate()

    @property
    def correl_matrix(self):
        return np.array(self.__correl, copy=True)

    @correl_matrix.setter
    def correl_matrix(self, val):
        if val is None:
            self.__correl = [[1]]
            return

        if not isinstance(val, np.ndarray):
            correl_matrix = np.array(val)

        if correl_matrix.ndim != 2:
            raise ValueError("The correl_matrix must be 2-dimensional")

        dim0, dim1 = correl_matrix.shape
        if dim0 != dim1 and self.dimension != dim0:
            raise ValueError("The correl_matrix is not square or doesn't match the dimension")

        if (correl_matrix.T != correl_matrix).any():
            raise ValueError("The correl_matrix is not symmetric")

        if correl_matrix.min() < -1 or correl_matrix.max() > 1:
            raise ValueError("The correl must lie between -1 and +1")

        self.__correl = correl_matrix

    @property
    def drifts(self):
        return np.array(self.__drifts, copy=True)

    @property
    def vols(self):
        return np.array(self.__vols, copy=True)

    def simulate(self):
        gauss = np.random.multivariate_normal(np.zeros(self.dimension), self.correl_matrix, self.time.size-1).T

        vol = np.multiply(self.vols, np.multiply(gauss, self._sqrt_delta_time))
        drift = np.multiply(self.drifts, self._delta_time)

        dXt = drift + vol
        vals = np.cumsum(np.hstack((self._x0, dXt)), 1)
        self.values = vals

    def _current_and_drift_integral(self, t, T):
        if t > T:
            return self(T), 0.

        if T not in self.time:
            raise ValueError("T must be in the time_grid, given: %s.\n Time grid = %s"%(T, self.time))

        current = self(t) # vector

        drift = self.__drifts
        integral = None
        if drift.shape[1] == 1:
            integral = (T-t) * drift
        else:
            index_t = np.where(self.time == t)[0][0]
            index_T = np.where(self.time == T)[0][0]
            
            time = self._delta_time[0, index_t:index_T]
            drift = drift[:, index_t:index_T]

            integral = np.zeros((self.dimension, 1))
            for i, d in enumerate(drift):
                integral[i] = np.dot(time, d)

        return current, integral

    def conditional_expectation(self, T, t):
        current, int = self._current_and_drift_integral(t, T)
        return current + int

    @staticmethod
    def compute_next_value(current, drift, vol, delta_t, gaussian_rv):
        dim = 1 if isinstance(current, float) else current.shape[0]
        delta_time = np.tile(delta_t, (dim, 1))
        sqrt_delta_time = np.sqrt(delta_time)

        dXt = np.multiply(drift, delta_time)
        dXt += np.multiply(np.multiply(vol, sqrt_delta_time), gaussian_rv)

        return current + dXt

Process.register(BrownianMotion)


class GeometricBrownianMotion(BrownianMotion):
    def conditional_expectation(self, T, t):
        current, int = self._current_and_drift_integral(t, T)
        return current * np.exp(int)

    def simulate(self):
        gauss = np.random.multivariate_normal(np.zeros(self.dimension), self.correl_matrix, self.time.size-1).T

        log_vol = np.multiply(self.vols, np.multiply(gauss, self._sqrt_delta_time))
        vol2 = np.square(self.vols)

        log_drift = np.multiply(self.drifts - 0.5*vol2, self._delta_time)

        dlogXt = log_drift + log_vol
        dXt = np.exp(dlogXt)

        self.values = np.cumprod(np.hstack((self._x0, dXt)), 1)

    @staticmethod
    def compute_next_value(current, drift, vol, delta_t, gaussian_rv):
        dim = 1 if isinstance(current, float) else current.shape[0]
        delta_time = np.tile(delta_t, (dim, 1))
        sqrt_delta_time = np.sqrt(delta_time)

        log_vol = np.multiply(vol, np.multiply(gaussian_rv, sqrt_delta_time))
        vol2 = np.square(vol)

        log_drift = np.multiply(drift - 0.5*vol2, delta_time)

        dlogXt = log_drift + log_vol
        dXt = np.exp(dlogXt)

        return np.multiply(current, dXt)

Process.register(GeometricBrownianMotion)