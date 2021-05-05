# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time

class CheckmarkModel(AbstractCostingFidelity):
    """
    Supported option keys in dict:


    """
    def __init__(self, nscenarios):
        AbstractCostingFidelity.__init__()
        self.default_options_dict = {}

    def compute_scenario_cost(self, reqired_arg_dict={}):
        # based on n_periods, compute cost_n, rolling window sum (if n_periods is 1, this will be the same as cost_1)
        cost_n = cost_1.rolling(n_periods).sum().shift(-(n_periods - 1))
        # rolling window operation looses digits, have to round (so we don't have negative values when adding zeroes)
        cost_n = cost_n.round(self.WTK_DATA_PRECISION)
        if (cost_n < 0).any():
            print("any neg values in cost_n? {}".format((cost_n < 0).any()))
        # IS
        # probability mass function g(s) i.e. importance distribution
        importance_probs = cost_n.loc[p_bin.index] / cost_n.loc[p_bin.index].sum()

        # sample of n_scenarios timestamps that are in wanted bin with probabilities given by cost_n series
        sample_timestamps = p_bin.sample(
            n_scenarios, random_state=random_seed, weights=importance_probs
        ).index

