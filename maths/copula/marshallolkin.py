# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 19:37:13 2015

@author: Yann
"""

import numpy as np
from scipy import special

class MarshallOlkinCopula(object):
    
    def __init__(self, reduced_index, total_number, indices, lambdas):
        assert reduced_index >= 0, "The reduced_index must be positive"
        self._reduced_index_ = reduced_index
        
        assert(reduced_index <= total_number), "The number of indexes is leq than the reduced_index"
        self._dimension_ = reduced_index
        
        if indices and lambdas:            
            self._subsets_ = np.array(indices)
            self._lambdas_ = np.array(lambdas)            
        else:
            raise NotImplementedError("Please give subsets of indexes and lambdas.")
        
        self._gamma_ = self.compute_gamma(self._reduced_index_)
        self._surv_subsets_ind_ = self.get_remaining_indexes_in_reduced_model(self._reduced_index_)
    
    def compute_gamma(self, reduced_index):
        gamma = 0
        for i in np.arange(self._subsets_.size):
            s = self._subsets_[i]
            if reduced_index in s:
                gamma += self._lambdas_[i]
                
        return gamma
    
    def get_remaining_indexes_in_reduced_model(self, reduced_index):
        res = []
        for i in np.arange(self._subsets_.size):
            s = self._subsets_[i]
            if reduced_index not in s:
                res.append(i)
                
        return np.array(res)
    
    def simulate_default_times(self, number=1, use_init_indexes=True, not_included_index=None):
        lambdas = self._lambdas_
        if use_init_indexes:
            lambdas = self._lambdas_[self._surv_subsets_ind_]
        else:
            if not_included_index:
                cp_subsets = self.get_counterparties_indices(not_included_index)
                lambdas = self._lambdas_[cp_subsets]
            else:
                raise ValueError("The not_included_index has not been given")
        
        U = np.random.uniform(size=(number, len(lambdas)))
        rvs = -np.log(U)/lambdas
        return rvs
        
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