from abc import abstractmethod, ABCMeta
import numpy as np
import sys


class AbsPortfolio(object):
    __metaclass__ = ABCMeta

    def __init__(self, matrix_positions, derivatives_notionals, derivatives, exposures):
        self.positions = matrix_positions

        self.notionals = derivatives_notionals
        self.derivatives = derivatives
        self.exposures = exposures

    @property
    def derivatives_number(self):
        return self.__positions.shape[1]

    @property
    def counterparties_number(self):
        return self.__positions.shape[0]

    @property
    def positions(self):
        return np.array(self.__positions, copy=True)

    @positions.setter
    def positions(self, value):
        val = value if isinstance(value, np.ndarray) else np.array(value)
        if val.ndim != 2:
            raise ValueError("The matrix value is not of dimension 2, given %s"%val.ndim)

        self.__positions = val

    def __check_derivatives_nb(self, value):
        val = value if isinstance(value, np.ndarray) else np.array(value)
        if val.ndim != 1:
            raise ValueError("The value is a matrix of dimension %s. It must be an array."%val.ndim)

        if val.shape[0] != self.derivatives_number:
            raise ValueError("The value has not the same dimension (%s) of the nb of derivatives %s"%(val.shape[0], self.derivatives_number))

        return val

    @property
    def notionals(self):
        return np.array(self.__notionals, copy=True)

    @notionals.setter
    def notionals(self, value):
        val = self.__check_derivatives_nb(value)
        self.__notionals = val

    @property
    def derivatives(self):
        return np.array(self.__derivatives, copy=True)

    @derivatives.setter
    def derivatives(self, value):
        val = self.__check_derivatives_nb(value)
        self.__derivatives = val

    @property
    def exposures(self):
        return np.array(self.__exposures, copy=True)

    @exposures.setter
    def exposures(self, value):
        val = self.__check_derivatives_nb(value)
        self.__exposures = val

    @abstractmethod
    def compute_value(self, t, **kwargs):
        pass

    @abstractmethod
    def compute_exposure(self, t, **kwargs):
        pass


class CSAPortfolio(AbsPortfolio):
    __directions = [1, -1]

    def __init__(self, matrix_positions, derivatives_notionals, derivatives, exposures, bank_id):
        mat = matrix_positions if isinstance(matrix_positions, np.ndarray) else np.array(matrix_positions)

        sum_col = mat.sum(axis=0)
        for (i, sum_) in enumerate(sum_col):
            if sum_ > 1e-15:
                raise ValueError("The total portfolio composition is not neutral "
                                 "for index = %i, sum=%s" % (i, sum_))

        self.__bank_id = bank_id
        self.__own_positions = None

        super(CSAPortfolio, self).__init__(matrix_positions, derivatives_notionals, derivatives, exposures)

    def _compute_positions_matrix(self, pov_index):
        if self.__own_positions is None:
            positions = self.positions
            try:
                mod_positions = np.divide(positions, -positions[self.__bank_id])
            except RuntimeWarning:
                zero_position_indexes = np.where(positions[self.__bank_id] == 0)
                mod_positions[:, zero_position_indexes] = 0.

            self.__own_positions = mod_positions

        res = np.array(self.__own_positions, copy=True)
        if pov_index != self.__bank_id:
            res = -res

        return res

    @property
    def counterparties_positions_from_bank(self):
        return np.array(self.__own_positions, copy=True)

    def _project_result(self, result, **kwargs):
        from_ = kwargs["from_"]
        towards_ = kwargs.get("towards_", None)

        # Computation of losses towards everyone
        if towards_ is None:
            if from_ != self.__bank_id:
                raise ValueError("Must provide towards keyword.")
            return result

        # Handle the point of view
        # If different of the bank_id, must take the
        # values at from_'s point of view.
        #
        # In that case, we already multiplied the positions
        # of all members by -1 in the _compute_positions_matrix method
        if from_ != self.__bank_id:
            if towards_ != self.__bank_id:
                raise ValueError("The point of view has not been set")
            towards_ = kwargs["from_"]

        return result[towards_]

    def compute_value(self, t, **kwargs):
        prices = np.array([d.price(t) for d in self.derivatives])

        pov_index = kwargs.get("from_", None)
        mod_positions = self._compute_positions_matrix(pov_index)

        #. by notional
        mod_positions = np.multiply(mod_positions, self.notionals)

        #. by prices
        result = np.multiply(mod_positions, prices)

        result = self._project_result(result, **kwargs)

        return result

    def compute_exposure(self, t, **kwargs):
        pov_index = kwargs.get("from_", None)
        mod_positions = self._compute_positions_matrix(pov_index)

        exposures = np.zeros(self.positions.shape)
        for (ii, e) in enumerate(self.exposures):
            exposure = e(t=t, **kwargs)
            for d, exp in zip(self.__directions, exposure):
                temp = np.sign(mod_positions[:, ii]) == d
                exposures[temp, ii] = np.maximum(mod_positions[temp, ii] * exp, 0)

        exposures = np.multiply(exposures, self.notionals)

        exposures = self._project_result(exposures, **kwargs)

        return exposures


class CCPPortfolio(CSAPortfolio):
    def __init__(self, matrix_positions, derivatives_notionals, derivatives, exposures):
        mat = matrix_positions if isinstance(matrix_positions, np.ndarray) else np.array(matrix_positions)
        super(CCPPortfolio, self).__init__(-mat, derivatives_notionals, derivatives, exposures, None)

    def _compute_positions_matrix(self, pov_index):
        return self.positions

    def _project_result(self, result, **kwargs):
        if "towards_" in kwargs:
            result = result[kwargs["towards_"]]

        return result

AbsPortfolio.register(CSAPortfolio)