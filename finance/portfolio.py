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
    __own_positions = None

    def __init__(self, matrix_positions, derivatives_notionals, derivatives, exposures, bank_id):
        mat = matrix_positions if isinstance(matrix_positions, np.ndarray) else np.array(matrix_positions)

        sum_col = mat.sum(axis=0)
        for (i, sum_) in enumerate(sum_col):
            if sum_ > 1e-15:
                raise ValueError("The total portfolio composition is not neutral "
                                 "for index = %i, sum=%s" % (i, sum_))

        self.__bank_id = bank_id

        self.__exposures_dict = {}
        self.__port_dict = {}

        super(CSAPortfolio, self).__init__(matrix_positions, derivatives_notionals, derivatives, exposures)

        self._compute_positions_matrix()

    def _compute_positions_matrix(self):
        positions = self.positions
        try:
            mod_positions = np.divide(positions, -positions[self.__bank_id])
        except RuntimeWarning:
            zero_position_indexes = np.where(positions[self.__bank_id] == 0)
            mod_positions[:, zero_position_indexes] = 0.

        self.__own_positions = mod_positions

    @property
    def counterparties_positions_from_self(self):
        return np.array(self.__own_positions, copy=True)

    def _get_weights(self, **kwargs):
        _from = kwargs["from_"]
        _towards = kwargs["towards_"]

        if _towards == _from:
            if _towards == self.__bank_id:
                return np.zeros((1, self.derivatives_number))
            raise ValueError("Can't have same from_ and towards_ keyword args")

        if _from == self.__bank_id:
            multiplier = 1.
        else:
            if _towards != self.__bank_id:
                raise ValueError("Neither from_ nor towards_ is equal to bank_id")
            multiplier = -1.
            _towards = _from

        weights = multiplier * np.array(self.__own_positions, copy=True)[_towards]

        return weights

    def compute_value(self, t, **kwargs):
        if t not in self.__port_dict:
            pricer = lambda prod: prod.price(t, **kwargs)
            self.__port_dict[t] = np.array(map(pricer, self.derivatives))

        prices = self.__port_dict[t]
        prices_by_not = np.multiply(self.notionals, prices)

        weights = self._get_weights(**kwargs)
        return np.multiply(weights, prices_by_not)

    def compute_exposure(self, t, **kwargs):
        params = self.exposures[0].param_names
        key = (t, tuple([kwargs[p] for p in params]))
        if key not in self.__exposures_dict:
            exposure_pricer = lambda expos_obj: expos_obj(t=t, **kwargs)
            self.__exposures_dict[key] = np.array(map(exposure_pricer, self.exposures))

        exposures = self.__exposures_dict[key]
        weights = self._get_weights(**kwargs)

        not_by_weights = np.multiply(weights, self.notionals)
        not_by_weights = not_by_weights.reshape((self.derivatives_number, 1))

        results = np.multiply(exposures, not_by_weights)
        res = np.amax(results, axis=1)

        return res

    def delete_cache(self):
        self.__port_dict.clear()
        self.__exposures_dict.clear()

    @property
    def own_index(self):
        return self.__bank_id


class CCPPortfolio(CSAPortfolio):
    def __init__(self, matrix_positions, derivatives_notionals, derivatives, exposures):
        ccp_pos = -(matrix_positions if isinstance(matrix_positions, np.ndarray) else np.array(matrix_positions))
        super(CCPPortfolio, self).__init__(ccp_pos, derivatives_notionals, derivatives, exposures, -1)

    def _compute_positions_matrix(self):
        pass

    def _get_weights(self, **kwargs):
        _from = kwargs.get("from_", self.own_index)
        _towards = kwargs.get("towards_", self.own_index)

        if _towards == _from:
            raise ValueError("Can't have same from_ and towards_ keyword args")

        if _from == self.own_index:
            multiplier = 1.
        else:
            if _towards != self.own_index:
                raise ValueError("Neither from_ nor towards_ is equal to bank_id")
            multiplier = -1.
            _towards = _from

        weights = multiplier * self.positions[_towards]
        return weights


AbsPortfolio.register(CSAPortfolio)