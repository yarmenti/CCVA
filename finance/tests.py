# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:53:59 2015

@author: Yann
"""

def test_discount_factor():
    from discountfactor import ConstantRateDiscountFactor
    
    df = ConstantRateDiscountFactor(0.)
    assert df(0) == df(1), "Error in ConstantRateDiscountFactor"
    assert df(1000) == 1., "Error in ConstantRateDiscountFactor"
    
def test_portfolio():
    from portfolio import Portfolio
    
    positions = [[10, 5], [5, -3], [-15, -2]]
        