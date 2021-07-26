# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time
from powerscenarios.costs.abstract_fidelity import AbstractCostingFidelity

class CheckmarkModel(AbstractCostingFidelity):
    """
    Supported option keys in dict:


    """
    def __init__(self,
                 n_scenarios, # Number of scenarios we actually want in our final csv file
                 n_periods,
                 loss_of_load_cost,
                 spilled_wind_cost,
                 scenarios_df,
                 p_bin,
                 WTK_DATA_PRECISION=6):

        AbstractCostingFidelity.__init__(self,
                                         n_scenarios,
                                         n_periods,
                                         loss_of_load_cost,
                                         spilled_wind_cost,
                                         scenarios_df,
                                         p_bin,
                                         WTK_DATA_PRECISION=6)


    def compute_scenario_cost(self,
                              random_seed=np.random.randint(2 ** 31 - 1)):
        # compute costs of each 1-period scenario
        self.cost_1 = self.scenarios_df["Deviation"].apply(
            lambda dev: np.abs(self.loss_of_load_cost * dev)
            if dev < 0
            else np.abs(self.spilled_wind_cost * dev)
        )
        # print("any neg values in cost_1? {}".format((cost_1 < 0).any()))

        # based on n_periods, compute cost_n, rolling window sum (if n_periods is 1, this will be the same as cost_1)
        self.cost_n = self.cost_1.rolling(self.n_periods).sum().shift(-(self.n_periods - 1))
        # rolling window operation looses digits, have to round (so we don't have negative values when adding zeroes)
        self.cost_n = self.cost_n.round(self.WTK_DATA_PRECISION)
        if (self.cost_n < 0).any():
            print("any neg values in cost_n? {}".format((self.cost_n < 0).any()))

        return self.cost_n
