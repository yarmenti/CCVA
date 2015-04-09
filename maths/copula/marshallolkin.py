# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 19:37:13 2015

@author: Yann
"""

import numpy as np
from scipy import special
from scipy.stats import expon
from scipy.optimize import brentq
from credit.default_models import StepwiseConstantIntensity


class MarshallOlkinCopula(object):
    def __init__(self, reduced_index, total_number, indices, lambdas):
        assert reduced_index >= 0, "The reduced_index must be positive"
        self.__reduced_index = reduced_index
        
        assert(reduced_index <= total_number), "The number of indexes is leq than the reduced_index"
        self.__dimension = total_number
        
        if indices and lambdas:            
            self.__subsets = np.array(indices)
            self.__lambdas = np.array(lambdas)
        else:
            raise NotImplementedError("Please give subsets of indexes and lambdas.")
        
        self.__gamma = self.compute_gamma(self.__reduced_index)
        self.__surv_subsets_ind = self.get_remaining_indexes_in_reduced_model(self.__reduced_index)
    
    def compute_gamma(self, reduced_index):
        gamma = 0
        for i in np.arange(self.__subsets.size):
            s = self.__subsets[i]
            if reduced_index in s:
                gamma += self.__lambdas[i]
                
        return gamma
    
    def get_remaining_indexes_in_reduced_model(self, reduced_index):
        res = []
        for i in np.arange(self.__subsets.size):
            s = self.__subsets[i]
            if reduced_index not in s:
                res.append(i)
                
        return np.array(res)
    
    def simulate_default_times(self, number=1, use_init_indexes=True, not_included_index=None):
        lambdas = self.__lambdas
        if use_init_indexes:
            lambdas = self.defaultable_intensities
        else:
            if not_included_index:
                cp_subsets = self.get_counterparties_indices(not_included_index)
                lambdas = self.__lambdas[cp_subsets]
            else:
                raise ValueError("The not_included_index has not been given")
        
        U = np.random.uniform(size=(number, len(lambdas)))
        rvs = -np.log(U)/lambdas
        return rvs
        
    @property
    def defaultable_subsets(self):
        return np.array(self.__subsets[self.__surv_subsets_ind], copy=True)
        
    @property
    def defaultable_intensities(self):
        return np.array(self.__lambdas[self.__surv_subsets_ind], copy=True)
        
    @classmethod
    def generate_subsets_and_intensities(cls, dimension):
        assert dimension > 0, "The dimension must be a positive number"
        dim_range = np.arange(2, dimension+1)
                
        def _gen_sample_(i, n):
            coeff = int(special.binom(n, i))
            return np.random.random_integers(low=0, high=coeff)
        
        vgen_sample_ = np.vectorize(_gen_sample_)
        subsets_counts = vgen_sample_(dim_range, dimension)
        
        while (np.sum(subsets_counts) == 0):            
            subsets_counts = vgen_sample_(dim_range, dimension)
                
        z = zip(dim_range, subsets_counts)
        res = []
        hashs = []
        for t in z:
            if t[1] > 0:                
                for j in np.arange(t[1]):
                    tmp_ = frozenset(np.random.choice(dimension, t[0], replace=False))
                    hash_ = hash(tmp_)
                    while hash_ in hashs:
                        tmp_ = frozenset(np.random.choice(dimension, t[0], replace=False))
                        hash_ = hash(tmp_)                        
                    res.append(tmp_) 
                    hashs.append(hash_)        
        
        res = np.array(res)
        
        lambdas_mo_h1_ind = np.arange(res.size)         
        # From Crepey => simregr-jumps-SUBMITTED
        vlambdas_mo_h1_fun = np.vectorize(lambda i: 2*0.001/(1+i))
        lambdas_mo_h1 = vlambdas_mo_h1_fun(lambdas_mo_h1_ind)
                
        one_dim_range = np.arange(dimension)
        vfrozenset = np.vectorize(lambda i: frozenset([i]))        
        one_dim_set = vfrozenset(one_dim_range)        
        
        lambdas_mo_eq1_ind = np.arange(dimension)
        # From Crepey => simregr-jumps-SUBMITTED
        vlambdas_mo_eq1_fun = np.vectorize(lambda i: 0.0001*(200-i))
        lambdas_mo_eq1 = vlambdas_mo_eq1_fun(lambdas_mo_eq1_ind)
        
        subsets = np.append(res, one_dim_set)
        lambdas = np.append(lambdas_mo_h1, lambdas_mo_eq1)
        
        return subsets, lambdas


class StepWiseIntensitiesMarshallOlkinCopula(MarshallOlkinCopula):
    def __init__(self, subsets, hazard_rates, pillars):
        self.__subsets = np.array(subsets)
        self.__hzrd_rates_mat = np.array(hazard_rates)
        self.__pills = np.array(pillars)

        if self.__hzrd_rates_mat.shape[0] == 1:
            self.__hzrd_rates_mat = self.__hzrd_rates_mat.T

        if self.__pills.ndim != 1:
            raise ValueError("The pillars dimension number must be 1")

        if self.__subsets.ndim != 1:
            raise ValueError("The subsets dimension number must be 1")

        nb_subsets, nb_pills = self.__hzrd_rates_mat.shape

        if nb_subsets != self.__subsets.shape[0]:
            raise ValueError("The subsets size must be the same as the hazard rates")

        if nb_pills != self.__pills.shape[0]:
            raise ValueError("The pillars size must be the same as the hazard rates")

        self.__models = [StepwiseConstantIntensity(self.pillars, hz) for hz in self.__hzrd_rates_mat]

    @property
    def subsets(self):
        return np.array(self.__subsets, copy=True)

    @property
    def intensities(self):
        return np.array(self.__hzrd_rates_mat, copy=True)

    @property
    def pillars(self):
        return np.array(self.__pills, copy=True)

    @property
    def models(self):
        return np.array(self.__models, copy=True)

    def get_indexes_including(self, index):
        res = []
        for (ii, set) in enumerate(self.subsets):
            if index in set:
                res.append(ii)

        return res

    @staticmethod
    def __objective(models, exp_rvs):
        res = []

        for m, e in zip(models, exp_rvs):
            f = lambda t: e + m.log_survival_proba(t)
            tau = brentq(f, 0, 10000)
            res.append(tau)

        return np.array(res)

    def generate_default_times(self, obligor_index, number=1, exp_rvs=None):
        indexes = self.get_indexes_including(obligor_index)

        models = self.models[indexes]

        if exp_rvs is not None:
            rvs = np.array(exp_rvs)
            if rvs.shape[1] != len(indexes):
                raise ValueError("The exp rv don't have the good columns size. "
                                 "Given %s, expected %s"%(rvs.shape[1], len(indexes)))
        else:
            rvs = expon.rvs(size=[number, len(indexes)])

        res = np.zeros(rvs.shape)
        for ii, exps in enumerate(rvs):
            res[ii, :] = self.__objective(models, exps)

        return res