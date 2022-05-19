# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time
from powerscenarios.costs.abstract_fidelity import AbstractCostingFidelity

#####################################################################################
####### this way is much slower (pricing scenarios one by one in p_bin)
#####################################################################################

class CheckmarkModel(AbstractCostingFidelity):
    """
    Supported option keys in dict:


    """

    def __init__(
        self,
        n_scenarios,  # Number of scenarios we actually want in our final csv file
        n_periods,
        loss_of_load_cost,
        spilled_wind_cost,
        scenarios_df,
        p_bin,
        total_power_t0,
        WTK_DATA_PRECISION=6,
    ):

        AbstractCostingFidelity.__init__(
            self,
            n_scenarios,
            n_periods,
            loss_of_load_cost,
            spilled_wind_cost,
            scenarios_df,
            p_bin,
            total_power_t0,
            WTK_DATA_PRECISION=6,
        )

    def checkmark_cost(self, deviation):
        if deviation < 0:
            cost = self.loss_of_load_cost * abs(deviation)
        else:
            cost = self.spilled_wind_cost * abs(deviation)
        return cost

    def compute_scenario_cost(self, random_seed=np.random.randint(2 ** 31 - 1)):

        p_bin = self.p_bin
        total_power_t0 = self.total_power_t0
        n_periods = self.n_periods
        checkmark_cost = self.checkmark_cost
        scenarios_df = self.scenarios_df

        cost_n = pd.Series(index=p_bin.index, dtype="float64")

        for scenario_start_t in p_bin.index[:]:
            ### note: single period scenario starts and ends with scenario_start_t
            scenario_cost = 0
            #### consecutive time periods: 1,2,3,...
            for period_timestamp in pd.date_range(
                scenario_start_t,
                scenario_start_t + pd.Timedelta("5min") * (n_periods - 1),
                freq="5min",
            ):
                deviation = scenarios_df.loc[period_timestamp, "Deviation"]
                scenario_cost += checkmark_cost(deviation)
                #print("deviation = {}".format(deviation))
                #print("deviation cost = {}".format(checkmark_cost(deviation)))
                #print("scenario_cost = {}".format(scenario_cost))

            #### record scenario cost to cost_n series 
            cost_n.loc[scenario_start_t] = scenario_cost

        return cost_n


# #########################################################################################
# ####### this way is much faster (pricing all scenarios in scenarios_df using rolling sum)
# ####### but t0 to t1 pricing needs revisiting
# #########################################################################################

# class CheckmarkModel(AbstractCostingFidelity):
#     """
#     Supported option keys in dict:


#     """
#     def __init__(self,
#                  n_scenarios, # Number of scenarios we actually want in our final csv file
#                  n_periods,
#                  loss_of_load_cost,
#                  spilled_wind_cost,
#                  scenarios_df,
#                  p_bin,
#                  total_power_t0,
#                  WTK_DATA_PRECISION=6):

#         AbstractCostingFidelity.__init__(self,
#                                          n_scenarios,
#                                          n_periods,
#                                          loss_of_load_cost,
#                                          spilled_wind_cost,
#                                          scenarios_df,
#                                          p_bin,
#                                          total_power_t0,
#                                          WTK_DATA_PRECISION=6)


#     def compute_scenario_cost(self,
#                               random_seed=np.random.randint(2 ** 31 - 1)):

#         scenarios_df = self.scenarios_df
#         loss_of_load_cost = self.loss_of_load_cost
#         spilled_wind_cost = self.spilled_wind_cost
#         n_periods = self.n_periods
#         p_bin = self.p_bin

#         # compute costs of each 1-period scenario
#         cost_1 = scenarios_df["Deviation"].apply(
#             lambda dev: np.abs(loss_of_load_cost * dev)
#             if dev < 0
#             else np.abs(spilled_wind_cost * dev)
#         )
#         # print("any neg values in cost_1? {}".format((cost_1 < 0).any()))

#         # based on n_periods, compute cost_n, rolling window sum (if n_periods is 1, this will be the same as cost_1)
#         cost_n = cost_1.rolling(n_periods).sum().shift(-(n_periods - 1))
#         # rolling window operation looses digits, have to round (so we don't have negative values when adding zeroes)
#         cost_n = cost_n.round(self.WTK_DATA_PRECISION)

#         if (cost_n < 0).any():
#             print("any neg values in cost_n? {}".format((cost_n < 0).any()))

#         # select only p_bin indeces
#         cost_n = cost_n.loc[p_bin.index]

#         return cost_n                                  




