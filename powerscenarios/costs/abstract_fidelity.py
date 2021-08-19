# abstract_fidelity.py
# The following file will contain classes pertaining to different models used
# for costing scenarios. Please this inherit this abstract class when creating
# classes that define the various fidelitities used for costing scenarios
import numpy as np
import pandas as pd
import os, sys, time

class AbstractCostingFidelity(object):
    def __init__(self,
                 n_scenarios,
                 n_periods,
                 loss_of_load_cost,
                 spilled_wind_cost,
                 scenarios_df,
                 p_bin,
                 total_power_t0,
                 WTK_DATA_PRECISION=6):

        self.n_scenarios = n_scenarios
        self.n_periods = n_periods
        self.loss_of_load_cost = loss_of_load_cost
        self.spilled_wind_cost = spilled_wind_cost
        self.scenarios_df = scenarios_df
        self.p_bin = p_bin
        self.total_power_t0 = total_power_t0
        self.WTK_DATA_PRECISION=WTK_DATA_PRECISION
