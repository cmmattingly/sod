# Psuedocode Implementation of Copula-Based Outlier Detection:
# https://arxiv.org/pdf/2009.09463.pdf

import numpy as numpy
import math
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import skew

class COPOD():
    
    def ecdf(self, X):
        ecdf = ECDF(X)
        return ecdf(X)
    
    def decision_function(self, X):
        # loop through features
        for feature in X.columns:
            # fit left tail ecdf
            # fit right tail ecdf
            # compute skewness coefficient for current dimension
            assert(col)

        # allocate memory for outlier scores
        outlier_scores = np.empty(len(X), dtype=float32)

        for i, row in enumerate(X):
            # compute empirical copula observations (vector of d values) 
            left_tail_values = None
            right_tail_values = None
            skew_values = None

            # calculate tail probabilities
            left_tail_prob = -math.log(left_tail_values)
            right_tail_prob = -math.log(right_tail_values)
            skew_prob = -math.log(skew_values)

            # calculate outlier score
            outlier_score = max([left_tail_prob, right_tail_prob, skew_prob])
            outlier_scores[i] = outlier_score

        return outlier_scores