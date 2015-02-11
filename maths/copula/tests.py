# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 19:53:26 2015

@author: Yann
"""

from marshallolkin import MarshallOlkinCopula

def test_init_copula():
    nb_cm = 3
    cm_index = 0

    mo_groups = [frozenset([i]) for i in range(nb_cm)]
    mo_groups.append(frozenset([0, 1]))
    mo_groups.append(frozenset([1, 2]))
    lambdas = [0.02, 0.02, 0.02, 0.002, 0.05]

    copula = MarshallOlkinCopula(cm_index, nb_cm, mo_groups, lambdas)
    
    assert copula.compute_gamma(0) == 0.022, "Test failed"
    
    #mo_groups = [ [0], [1], [2], [0, 1], [1, 2] ]    
    assert (copula.get_remaining_indexes_in_reduced_model(0) == [1, 2, 4]).all(), "Test failed"
    assert (copula.get_remaining_indexes_in_reduced_model(1) == [0, 2]).all(), "Test failed"
    assert (copula.get_remaining_indexes_in_reduced_model(2) == [0, 1, 3]).all(), "Test failed"
    
def test_generate_subsets_and_lambdas():
    a, b = MarshallOlkinCopula.generate_subsets_and_intensities(5)
    
    assert a.size == b.size
    
    print a
    print b