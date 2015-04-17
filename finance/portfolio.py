# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:55:50 2015

@author: Yann
"""

import numpy as np
import sys


class Portfolio(object):

    @staticmethod
    def generate_1_vs_all_positions(alone_index, positive_single_pos_sgn, obligors_nb):
        if alone_index >= obligors_nb:
            raise ValueError("alone_index = %s must be leq than "
                             "obligors_nb = %s"%(alone_index, obligors_nb))

        sgn = 1. if positive_single_pos_sgn else -1.

        weights = -sgn/(obligors_nb-1)
        res = np.empty(obligors_nb)
        res.fill(weights)
        res[alone_index] = sgn

        return res

    def __init__(self, matrix_positions, derivatives, exposures):
        self.positions = matrix_positions
        self.derivatives = derivatives
        self.exposures = exposures

    @property
    def bank_numbers(self):
        return self.positions.shape[0]

    @property
    def positions(self):
        return np.array(self.__positions, copy=True)

    @positions.setter
    def positions(self, value):
        val = value if isinstance(value, np.ndarray) else np.array(value)
        if val.ndim != 2:
            raise ValueError("The matrix value is not of dimension 2, given %s"%val.ndim)
        self.__positions = val
        self.__nb_der = self.__positions.shape[1]
        self.__init_new_positions()

    def __init_new_positions(self):
        self.__notionals = np.absolute(self.__positions)
        amounts = self.notionals.sum(axis=1)
        self.__weights = (self.notionals.T / amounts).T
        self.__directions = np.sign(self.__positions)

    @property
    def weights(self):
        return np.array(self.__weights, copy=True)

    @property
    def notionals(self):
        return np.array(self.__notionals, copy=True)

    @property
    def directions(self):
        return np.array(self.__directions, copy=True)

    @property
    def derivatives(self):
        return np.array(self.__derivatives, copy=True)

    @derivatives.setter
    def derivatives(self, value):
        der = np.array(value)
        if der.ndim != 1:
            raise ValueError("The dimension of the derivatives must be 1, given %s"%der.ndim)
        if der.size != self.__nb_der:
            raise ValueError("The derivatives size must be equal to %s, given: %s"%(self.__nb_der, der.size))

        self.__derivatives = der

    @property
    def exposures(self):
        return np.array(self.__exposures, copy=True)
    
    @exposures.setter
    def exposures(self, value):
        expos = np.array(value)
        if expos.ndim != 1:
            raise ValueError("The dimension of the exposures must be 1, given %s"%expos.ndim)
        if expos.size != self.__nb_der:
            raise ValueError("The exposures size must be equal to %s, given: %s"%(self.__nb_der, expos.size))

        self.__exposures = expos

    def compute_value(self, derivative_prices, **kwargs):
        res = np.dot(self.positions, derivative_prices).reshape((self.bank_numbers, 1))
        if 'from_' in kwargs:
            res = res[kwargs['from_'], :]
            if 'towards_' in kwargs:
                projection = self.compute_projection(kwargs['from_'], kwargs['towards_'])
                res = np.multiply(res, projection)

        return res

    def compute_exposure(self, t, **kwargs):
        directions = [1, -1]
        exposures = np.zeros(self.positions.shape)
        for (ii, e) in enumerate(self.exposures):
            exposure = e(t=t, **kwargs)
            for d, exp in zip(directions, exposure):
                temp = self.directions[:, ii] == d
                exposures[:, ii][temp] = np.maximum(self.positions[:, ii][temp] * exp, 0)

        if 'from_' in kwargs:
            exposures = exposures[kwargs['from_'], :]
            if 'towards_' in kwargs:
                projection = self.compute_projection(kwargs['from_'], kwargs['towards_'])
                exposures = np.multiply(exposures, projection)
                if kwargs.get('total', False):
                    exposures = exposures.sum()

        return exposures

    def compute_projection(self, from_, towards_):
        raise NotImplementedError("Must be implemented in a subclass")

class EquilibratedPortfolio(Portfolio):
    def __init__(self, matrix_positions, derivatives, exposures):
        mat = matrix_positions if isinstance(matrix_positions, np.ndarray) else np.array(matrix_positions)

        sum_col = mat.sum(axis=0)
        for (i, sum_) in enumerate(sum_col):
            if sum_ > sys.float_info.epsilon:
                raise ValueError("The total portfolio composition is not neutral "
                                 "for index = %i, sum=%s"%(i, sum_))

        super(EquilibratedPortfolio, self).__init__(matrix_positions, derivatives, exposures)

    @property
    def positions(self):
        return super(EquilibratedPortfolio, self).positions

    @positions.setter
    def positions(self, value):
        super(EquilibratedPortfolio, self.__class__).positions.fset(self, value)
        self.__weights_asset = np.zeros(self.positions.shape)
        for ii in range(self.__weights_asset.shape[1]):
            col = self.positions[:, ii]
            pos_idx = col>0
            neg_idx = col<0
            pos = float(np.array(col[pos_idx]).sum())

            self.__weights_asset[:, ii][pos_idx] = col[pos_idx] / pos
            self.__weights_asset[:, ii][neg_idx] = col[neg_idx] / pos

    def compute_projection(self, from_, towards_):
        if from_ >= self.bank_numbers or towards_ >= self.bank_numbers:
            raise ValueError("The indexes must be less than %s"%self.bank_numbers)

        if from_ == towards_:
            return np.zeros(self.__weights_asset.shape[1])

        w_from = self.__weights_asset[from_, :]
        w_towards = np.array(self.__weights_asset[towards_, :], copy=True)

        prod = np.multiply(w_from, w_towards)
        for (i, p) in enumerate(prod):
            if p >= 0:
                w_towards[i] = 0.

        return np.absolute(w_towards)

class CCPPortfolio(EquilibratedPortfolio):
    def __init__(self, members_positions_mat, derivatives, exposures):
        ccp_positions = -members_positions_mat
        super(CCPPortfolio, self).__init__(ccp_positions, derivatives, exposures)