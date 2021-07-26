# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time

class ExaGO_File(AbstractCostingFidelity):
    """
    This class contains the wrapper for calling OPFLOW from within Powerscenaios
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
