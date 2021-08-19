from __future__ import print_function
import logging
import pandas as pd
import numpy as np
import sys

# should this be imported only as needed (in retrieve_wind_sites, retrieve_wtk_data,)
import os

from powerscenarios.costs.checkmark import CheckmarkModel
#from powerscenarios.costs.exago import ExaGO_File

logging.basicConfig()

# do this before importing pywtk, so that WIND_MET_NC_DIR and WIND_FCST_DIR are set correctly
# if cache dir is set, will use AWS (as opposed to local data)
#os.environ["PYWTK_CACHE_DIR"] = os.path.join(os.environ["HOME"], "pywtk-data")

from pywtk.wtk_api import get_nc_data, site_from_cache, WIND_FCST_DIR, WIND_MET_NC_DIR
import pywtk


class Grid(object):
    """ docstring TBD
    """

    def __init__(
        self,
        name,
        buses,
        generators,
        wind_generators,
        wind_sites=pd.DataFrame(),
        actuals=None,
        scenarios=None,
    ):
        self.name = name
        self.buses = buses
        self.generators = generators
        self.wind_generators = wind_generators
        self.wind_sites = wind_sites
        self.actuals = actuals
        self.scenarios = scenarios

    # attributes

    blet = "blet"
    WTK_DATA_PRECISION = 6

    # if __repr__ is defined, then __str__ = __repr__ (converse is not true)
    #    def __str__(self):
    #        return 'Grid(name='+self.name+', buses='+str(len(self.buses)) + ', generators='+str(len(self.generators)) + ', wind_generators='+str(len(self.wind_generators)) + ')'

    def __repr__(self):
        #        return self.name + ' grid object: buses='+str(len(self.buses)) + ' generators='+str(len(self.generators)) + ' wind generators='+str(len(self.wind_generators))
        if self.wind_sites is None:
            n_wind_sites = "None"
        else:
            n_wind_sites = str(len(self.wind_sites))

        return (
            "Grid(name="
            + self.name
            + ", buses="
            + str(len(self.buses))
            + ", generators="
            + str(len(self.generators))
            + ", wind_generators="
            + str(len(self.wind_generators))
            + ", wind_sites="
            + n_wind_sites
            + ")"
        )

    def info(self):
        """ Method for displaying grid statistics
        """

        info_str = ""

        info_str = info_str + "\n{} grid info: \n".format(self.name)

        # number of busses:
        n_buses = len(self.buses)
        info_str = info_str + "\n number of buses: {}".format(n_buses)

        # number of generators
        n_generators = len(self.generators)
        info_str = info_str + "\n number of generators: {}".format(n_generators)

        # number of wind generators
        n_wind_generators = len(self.wind_generators)
        info_str = info_str + "\n number of wind generators: {}".format(
            n_wind_generators
        )

        # number of solar generators
        n_solar_generators = len(
            self.generators[self.generators["GenFuelType"] == "Solar"]
        )
        info_str = info_str + "\n number of solar generators: {}".format(
            n_solar_generators
        )

        # total capacity
        total_capacity = self.generators["GenMWMax"].sum()
        info_str = info_str + "\n total generator capacity: {:.2f} MW".format(
            total_capacity
        )

        # wind capacity
        wind_capacity = self.generators[self.generators["GenFuelType"] == "Wind"][
            "GenMWMax"
        ].sum()
        wind_penetration = 100 * wind_capacity / total_capacity
        info_str = (
            info_str
            + "\n wind capacity/penetration: {:.2f} MW / {:.2f}%".format(
                wind_capacity, wind_penetration
            )
        )

        # solar capacity
        solar_capacity = self.generators[self.generators["GenFuelType"] == "Solar"][
            "GenMWMax"
        ].sum()
        solar_penetration = 100 * solar_capacity / total_capacity
        info_str = (
            info_str
            + "\n solar capacity/penetration: {:.2f} MW / {:.2f}%".format(
                solar_capacity, solar_penetration
            )
        )

        return info_str

    def change_wind_penetration(
        self, new_wind_penetration, **kwargs,
    ):
        """ Method to change wind penetration (multiplying existing wind generator capacities by a factor)
            Changes generators and wind_generators.
        Required Arg:
            new_wind_penetration -  (float), new wind penetration as percentage of total capacity, [0,100)
        """

        # print('OLD:')
        # total capacity
        total_capacity = self.generators["GenMWMax"].sum()
        # print('total gen capacity: {}'.format(total_capacity))
        # wind capacity
        wind_capacity = self.generators[self.generators["GenFuelType"] == "Wind"][
            "GenMWMax"
        ].sum()
        # print('total wind capacity: {}'.format(wind_capacity))

        # wind penetration
        wind_penetration = 100 * wind_capacity / total_capacity
        # print('curent wind penetration: {:.2f}%'.format(wind_penetration))

        # now find a factor by which to multiply wind capacity to get new_wind_penetration
        print("\n\nChanging wind penetration to: {:.2f}%".format(new_wind_penetration))
        factor = (
            new_wind_penetration
            * (wind_capacity - total_capacity)
            / (wind_capacity * (new_wind_penetration - 100))
        )
        print("using factor of {}".format(factor))

        # modify GenMWMax column of wind_gen_df to achieve desired penetration
        self.wind_generators["GenMWMax"] = self.wind_generators["GenMWMax"] * factor

        # also need to modify gen_df to be consistent
        self.generators.loc[self.generators["GenFuelType"] == "Wind", "GenMWMax"] = (
            self.generators[self.generators["GenFuelType"] == "Wind"]["GenMWMax"]
            * factor
        )

    #         print('NEW:')
    #         total_capacity = self.generators['GenMWMax'].sum()
    #         print('total gen capacity: {}'.format(total_capacity))
    #         wind_capacity = self.generators[self.generators['GenFuelType']=='Wind']['GenMWMax'].sum()
    #         print('total wind capacity: {}'.format(wind_capacity))
    #         wind_penetration = 100*wind_capacity/total_capacity
    #         print('curent wind penetration: {:.2f}%'.format(wind_penetration))

    def retrieve_wind_sites(self, method = "simple proximity", **kwargs):
        """ Method to retrieve wind sites (SiteID) nearest to wind generators (up to their capacity, GenMWMax).
            Requires pywtk_api.

         Required Args:
            method='simple proximity',  TODO: 'capacity factor'

        """

        # add Point (wkt) column to wind_gen_df
        # i.e. turn Latitude and Longitude of wind generators into wkt POINT format (add it as a new column)
        # both, astype and apply work
        # wind_gen_df['wkt'] = 'POINT(' + wind_gen_df['Longitude'].astype(str) + ' ' + wind_gen_df['Latitude'].astype(str) + ')'
        if method == "simple proximity":

	        wind_gen_df = self.wind_generators

	        wind_gen_df["Point"] = (
	            "POINT("
	            + wind_gen_df["Longitude"].apply(str)
	            + " "
	            + wind_gen_df["Latitude"].apply(str)
	            + ")"
	        )

	        print("Retrieving wind sites ...")
	        # will create a DataFrame out of this list of dicts (rows)
	        wind_sites_list = []

	        site_ids_list = []  # for keeping track of used sites (don't want repeats)

	        for row in wind_gen_df.itertuples():
	            gen_capacity = row.GenMWMax
	            gen_wkt_point = row.Point
	            # retrieve wind sites sorted by proximity to gen location
	            sorted_sites = pywtk.site_lookup.get_3tiersites_from_wkt(gen_wkt_point)
	            # keep adding sites to the list until gen capacity is exceeded
	            total_sites_capacity = 0.0
	            for site in sorted_sites.itertuples():

	                wind_site = {
	                    "SiteID": site.Index,
	                    "Capacity": site.capacity,
	                    "Point": str(site.point),
	                    "Latitude": site.lat,
	                    "Longitude": site.lon,
	                    "BusNum": row.BusNum,  # add BusNum this site belongs to (maybe it'll be usefull)
	                    "GenUID": row.GenUID,  # add GenUID (generator unique ID) this site belongs to
	                }
	                # note that site.point is of type : shapely.geometry.point.Point
	                # hence, turn it into str, since get_wind_data_by_wkt() wants a str (stupid, isn't it?)

	                # if wind site is not in the list already, add it to the list (don't want repeats)
	                if not (site.Index in site_ids_list):
	                    wind_sites_list.append(wind_site)
	                    site_ids_list.append(site.Index)
	                    total_sites_capacity += site.capacity

	                if total_sites_capacity > gen_capacity:
	                    break

	        wind_sites_df = pd.DataFrame(wind_sites_list)

	        self.wind_sites = wind_sites_df
	        print("Done")
        # return wind_sites_df

    # internal, used for make_tables
    def retrieve_wtk_data(
        self, start_of_data, end_of_data, nc_dir="met", attributes=["power"], **kwargs,
    ):
        """ Function to retrieve wind power data using self.wind_sites
            Used to create initial actuals_df and scenarios_df

        Required Args:
            start_of_data - (pd.Timestamp) start of required power data
            end_of_data - (pd.Timestamp) end of required power data

        Optional Args:
            nc_dir - (string) either 'met' for meteorological (WIND_MET_NC_DIR) or 'fcst' for forecast (WIND_FCST_DIR)

        Returns:
            pd.DataFrame with columns as BusNum from wind_site_df and
            rows of power values indexed by Timestamp
            i.e. power data from wind sites is aggregated by the bus they belong to



        """

        if nc_dir == "met":
            nc_dir = WIND_MET_NC_DIR
        elif nc_dir == "fcst":
            nc_dir = WIND_FCST_DIR

        wind_sites_df = self.wind_sites
        if wind_sites_df.empty:
            raise Exception(
                "No wind sites, retrieve wind sites before retrieving data."
            )

            # print('No wind sites, retrieve wind sites before retrieving data.')
            # return

        # what do we want? - power!
        # attributes = ['power',]

        print("Retrieving WTK data ...")
        # initialize DataFrame by pulling one data point
        site_id = wind_sites_df["SiteID"].loc[0]
        wind_data_df_index = pywtk.wtk_api.get_nc_data(
            site_id,
            start_of_data,
            end_of_data,
            attributes=attributes,
            leap_day=True,
            utc=True,
            nc_dir=nc_dir,
        ).index

        wind_data_df = pd.DataFrame(index=wind_data_df_index)

        #  we need unique GenUID because BusNum is not unique ( multiple wind generators attached to one bus happen)
        # bus_numbers = wind_sites_df['BusNum'].unique()
        gen_uids = wind_sites_df["GenUID"].unique()
        # for bus_number in bus_numbers:
        for gen_uid in gen_uids:
            # actuals_df['Bus'+str(bus_number)] = 0.

            # initialize column
            # wind_data_df[bus_number] = 0.
            wind_data_df[gen_uid] = 0.0

            # site_ids = wind_sites_df[wind_sites_df['BusNum']== bus_number]['SiteID'].values
            # take site_ids that belong to the same gen_uid
            site_ids = wind_sites_df[wind_sites_df["GenUID"] == gen_uid][
                "SiteID"
            ].values
            for site_id in site_ids:
                # retrieve by site_id and keep adding
                wind_data_df_temp = pywtk.wtk_api.get_nc_data(
                    site_id,
                    start_of_data,
                    end_of_data,
                    attributes=attributes,
                    leap_day=True,
                    utc=True,
                    nc_dir=nc_dir,
                )

                # wind_data_df[bus_number]+=wind_data_df_temp[attributes[0]].values
                wind_data_df[gen_uid] += wind_data_df_temp[attributes[0]].values

        # add name for column index?
        # wind_data_df.columns.rename('BusNum',inplace=True)

        # rename row index
        wind_data_df.index.rename("IssueTime", inplace=True)

        print("Done")

        return wind_data_df

    # new method, this one does not power condition, makes actuals_df and scenarios_df
    def make_tables(
        self,
        actuals_start=pd.Timestamp("2007-01-01 00:00:00", tz="utc"),
        actuals_end=pd.Timestamp("2007-12-31 23:55:00", tz="utc"),
        scenarios_start=pd.Timestamp("2008-01-01 00:00:00", tz="utc"),
        scenarios_end=pd.Timestamp("2013-12-31 23:55:00", tz="utc"),
        **kwargs,
    ):
        """ Method retrieves data from wtk and makes actuals(DataFrame) and scenarios(DataFrame)
            Timestamps for actuals and scenarios: pick dates from 2007-01-01 to 2013-12-31
            Required args:
                actuals_start - (pd.Timestamp)
                actuals_end - (pd.Timestamp)
                scenarios_start - (pd.Timestamp)
                scenarios_end - (pd.Timestamp)
                source - (str) placeholder
        """

        # time window selection:
        # one year, e.g. 2007, for the actuals
        # start_of_data = pd.Timestamp('2007-01-01 00:00:00', tz='utc')
        # end_of_data = pd.Timestamp('2007-12-31 23:55:00', tz='utc')
        # start_of_data = pd.Timestamp('2007-01-01 00:00:00').tz_localize('US/Pacific')
        # end_of_data = pd.Timestamp('2007-12-31 23:55:00').tz_localize('US/Pacific')

        # one year, 2013, for the actuals
        # start_of_data = pd.Timestamp('2013-01-01 00:00:00', tz='utc')
        # end_of_data = pd.Timestamp('2013-12-31 23:55:00', tz='utc')

        wind_sites_df = self.wind_sites
        actuals_df = self.retrieve_wtk_data(actuals_start, actuals_end)

        # index does not have timezone in Devon's code, but it should
        # actuals_df.index = actuals_df.index.tz_localize(None)

        ###### fix "over" problem
        # max actual power can not go above GenMWMax
        # note that, it does not go "under" - WTK takes care of that
        max_gen_capacity = self.wind_generators.set_index("GenUID")["GenMWMax"]
        actuals_df = actuals_df[actuals_df < max_gen_capacity].fillna(max_gen_capacity)
        # grid.actuals = new_df

        # add total power column (accross all buses)
        actuals_df["TotalPower"] = actuals_df.sum(axis=1).values

        self.actuals = actuals_df

        # for scenarios_df, just change time window, last 6 years
        # start_of_data = pd.Timestamp('2008-01-01 00:00:00', tz='utc')
        # end_of_data = pd.Timestamp('2013-12-31 23:55:00', tz='utc')
        # start_of_data = pd.Timestamp('2008-01-01 00:00:00').tz_localize('US/Pacific')
        # end_of_data = pd.Timestamp('2013-12-31 23:55:00').tz_localize('US/Pacific')

        # for scenarios_df, just change time window, first 6 years
        # start_of_data = pd.Timestamp('2007-01-01 00:00:00', tz='utc')
        # end_of_data = pd.Timestamp('2012-12-31 23:55:00', tz='utc')

        scenarios_df = self.retrieve_wtk_data(scenarios_start, scenarios_end)

        # fix "over". same as for actuals_df, but different way
        # for GenMWMax info we take wind generators and reindex by GenUID
        wind_generators_df = self.wind_generators.set_index("GenUID")
        scenarios_df=scenarios_df.where(
            scenarios_df <= wind_generators_df["GenMWMax"],
            other=wind_generators_df["GenMWMax"],
            axis=1,
            #inplace=True,
        )

        # index does not have timezone in Devon's code, but it should
        # scenarios_df.index = scenarios_df.index.tz_localize(None)

        # add total power column (accross all generators)
        scenarios_df["TotalPower"] = scenarios_df.sum(axis=1).values

        # compute deviations
        # i.e. make error calculations (instead of full power at each bus, have deviations from persistence)

        # deviations from persistence at generators
        gen_deviations_array = (
            scenarios_df.iloc[1:, :-1].values - scenarios_df.iloc[:-1, :-1].values
        )
        # total power deviations
        total_power_deviations_array = (
            scenarios_df["TotalPower"].values[1:]
            - scenarios_df["TotalPower"].values[:-1]
        )
        # drop last row
        scenarios_df.drop(scenarios_df.tail(1).index, inplace=True)
        # record deviations
        scenarios_df.iloc[:, :-1] = gen_deviations_array
        scenarios_df["Deviation"] = total_power_deviations_array

        self.scenarios = scenarios_df



    def generate_wind_scenarios(
        self,
        timestamp,
        power_quantiles=[0.0, 0.1, 0.9, 1.0],
        sampling_method="monte carlo",
        fidelity="checkmark",
        n_scenarios=5,
        n_periods=1,
        random_seed=25,
        output_format=0,
        **kwargs,
    ):
        """Method to generate scenarios
	        Required Args:
	            timestamp - (pd.Timestamp)
	            power_quantiles - (list) quantiles for power conditioning
	            sampling_method - (string) either "importance" or "monte carlo"
	            n_scenarios - (integer) number of scenarios to draw
	            n_periods - (integer) number of periods for each scenario

	        Returns:
	            scenarios_df - (pd.DataFrame) multi-indexed DataFrame with all scenarios

	    """

        actuals_df = self.actuals
        scenarios_df = self.scenarios

        #         # TODO: basic checks:
        #         # still needs work to Raise ValueExceptions
        #         for timestamp in timestamps:
        #             # type
        #             if type(timestamp) != pd._libs.tslibs.timestamps.Timestamp:
        #                 print("timestamps must be pandas.Timestamp type")
        #                 return

        #         # t0 and t1 must be in actuals.index
        #         if not (timestamps[0] in actuals_df.index) or not (
        #             timestamps[1] in actuals_df.index
        #         ):
        #             print(
        #                 "timestamps[0] and timestamps[1] must be between {} and {}".format(
        #                     actuals_df.index.min(), actuals_df.index.max()
        #                 )
        #             )
        #             return

        #         # check if t0 < t1
        #         if not (timestamps[0] < timestamps[1]):
        #             print("timestamps[0] must be < timestamps[1]")
        #             return
        # all needed timestamps

        # all needed timestamps
        timestamps = pd.date_range(
            start=timestamp - pd.Timedelta("5min"), periods=n_periods + 1, freq="5min"
        )
        # print(timestamps)
        # power at t0
        total_power_t0 = actuals_df.loc[timestamps[0]]["TotalPower"]
        # total_power_t0

        # power conditioning
        # .iloc[:-n_periods] is so that we can find consecutive timestamps for multiperiod scenarios
        power_bins = pd.qcut(
            scenarios_df["TotalPower"].iloc[:-n_periods], q=power_quantiles,
        )
        # power_bins

        # which power bin does it belong to?
        # for power_bin in power_bins.unique():
        for power_bin in power_bins.cat.categories:
            if total_power_t0 in power_bin:
                break
        # power_bin
        # wanted power bin
        p_bin = power_bins.loc[power_bins == power_bin]

        if sampling_method == "monte carlo":
            # sample of n_scenarios timestamps that are in wanted bin
            sample_timestamps = p_bin.sample(n_scenarios, random_state=random_seed).index
            # #     # df of weights to return (all ones, redundant, but have to return something)
            weights_df = pd.DataFrame(index=[timestamps[1]], columns=range(1, n_scenarios + 1))
            weights_df.loc[timestamps[1]] = dict(
                zip(range(1, n_scenarios + 1), np.ones(n_scenarios))
            )


        elif sampling_method == "importance":

            # importance sample from an appropriate power bin
            # cost of MW per 5-min period
            loss_of_load_cost = 10000 / 12.0
            spilled_wind_cost = 0.001


            if fidelity == "checkmark":
                pmodel = CheckmarkModel(n_scenarios, # Number of scenarios we actually want in our final csv file
                                        n_periods,
                                        loss_of_load_cost,
                                        spilled_wind_cost,
                                        scenarios_df,
                                        p_bin,
                                        total_power_t0)
                cost_n = pmodel.compute_scenario_cost(random_seed=594081473)

            elif fidelity == "exago":
                pmodel = ExaGO_File(n_scenarios, # Number of scenarios we actually want in our final csv file
                                    n_periods,
                                    loss_of_load_cost,
                                    spilled_wind_cost,
                                    scenarios_df,
                                    p_bin)
                scenarios_df_copy = scenarios_df.drop(columns=["TotalPower", "Deviation"])
                cost_n = pmodel.compute_scenario_cost(actuals_df,
                                                      scenarios_df_copy,
                                                      timestamp,
                                                      random_seed=594081473)
            else:
                raise NotImplementedError
            # # compute costs of each 1-period scenario
            # cost_1 = scenarios_df["Deviation"].apply(
            #     lambda dev: np.abs(loss_of_load_cost * dev)
            #     if dev < 0
            #     else np.abs(spilled_wind_cost * dev)
            # )
            # # print("any neg values in cost_1? {}".format((cost_1 < 0).any()))
            #
            # # based on n_periods, compute cost_n, rolling window sum (if n_periods is 1, this will be the same as cost_1)
            # cost_n = cost_1.rolling(n_periods).sum().shift(-(n_periods - 1))
            # # rolling window operation looses digits, have to round (so we don't have negative values when adding zeroes)
            # cost_n = cost_n.round(self.WTK_DATA_PRECISION)
            # if (cost_n < 0).any():
            #     print("any neg values in cost_n? {}".format((cost_n < 0).any()))

            # IS
            # probability mass function g(s) i.e. importance distribution
            importance_probs = cost_n / cost_n.sum()

            # sample of n_scenarios timestamps that are in wanted bin with probabilities given by cost_n series
            sample_timestamps = p_bin.sample(
                n_scenarios, random_state=random_seed, weights=importance_probs
            ).index

            # IS weights: f(s)/g(s), i.e. nominal/importance
            importance_weights = (1 / p_bin.size) / importance_probs.loc[sample_timestamps]

            # df of weights to return
            weights_df = pd.DataFrame(index=[timestamps[1]], columns=range(1, n_scenarios + 1))
            weights_df.loc[timestamps[1]] = dict(
                zip(range(1, n_scenarios + 1), importance_weights.values)
            )

        # initialize multi-indexed df for all scenarios to return (one sim timestamp)
        iterables = [[timestamps[1]], range(1, n_scenarios + 1), timestamps[1:]]
        # index
        index = pd.MultiIndex.from_product(
            iterables, names=["sim_timestamp", "scenario_nr", "period_timestamp"]
        )
        # multi_scenarios_df
        multi_scenarios_df = pd.DataFrame(index=index, columns=actuals_df.columns[:-1])

        # now find wanted periods for each scenario i.e. consecutive timestamps in scenario_df
        for sample_i, sample_timestamp in enumerate(sample_timestamps):
            #print("sample_timestamp={}".format(sample_timestamp))
            # needed timestamps
            # pd.date_range(start=sample_timestamp, periods=n_periods, freq="5min")
            # deviation will be a df even with 1 period because loc is used with pd.date_range
            deviation_df = scenarios_df.loc[
                pd.date_range(start=sample_timestamp, periods=n_periods, freq="5min")
            ].copy()

            # change deviation index to match actuals we adding it to
            deviation_df.index = timestamps[1:]

            # make scenario df for each sample_timestamp (could be done outside the loop)
            scenario_df = pd.DataFrame(
                index=timestamps[1:], columns=deviation_df.columns
            ).drop(["Deviation", "TotalPower"], axis=1)
            #     print("\nscenario_df:")
            #     scenario_df

            #     print("\nactual:")
            #     actuals_df.loc[timestamps[0]]

            # first take actual
            running_sum = actuals_df.loc[timestamps[0]].drop("TotalPower").copy()
            # now keep adding deviations to the above actual
            for timestamp in timestamps[1:]:
                running_sum += deviation_df.loc[timestamp].drop(
                    ["TotalPower", "Deviation"],
                )
                scenario_df.loc[timestamp] = running_sum

            # "under/over" problem
            # for GenMWMax info we take wind generators and reindex by GenUID
            wind_generators_df = self.wind_generators.set_index("GenUID")
            # under
            # .where replaces False
            scenario_df.where(scenario_df >= 0.0, other=0.0, inplace=True)
            # over
            # scenario < wind_generators_df["GenMWMax"]
            scenario_df=scenario_df.where(
                scenario_df <= wind_generators_df["GenMWMax"],
                other=wind_generators_df["GenMWMax"],
                axis=1,
                #inplace=True,
            )
            # scenario

            #     # add each scenario to the multi-indexed df to return

            multi_scenarios_df.loc[(timestamps[1], sample_i + 1,)] = scenario_df.values
        return multi_scenarios_df, weights_df
