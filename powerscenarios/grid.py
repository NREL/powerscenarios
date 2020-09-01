from __future__ import print_function
import logging
import pandas as pd
import numpy as np
import sys

# should this be imported only as needed (in retrieve_wind_sites, retrieve_wtk_data,)
import os

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

    blet = "ble"
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

    def retrieve_wind_sites(self, **kwargs):
        """ Method to retrieve wind sites (SiteID) nearest to wind generators (up to their capacity, GenMWMax).
            Requires pywtk_api. 

        TODO: Required Args:
            method='nearest'  vs. 'capacity factor'

        """

        # add Point (wkt) column to wind_gen_df
        # i.e. turn Latitude and Longitude of wind generators into wkt POINT format (add it as a new column)
        # both, astype and apply work
        # wind_gen_df['wkt'] = 'POINT(' + wind_gen_df['Longitude'].astype(str) + ' ' + wind_gen_df['Latitude'].astype(str) + ')'

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

    # old make tables method
    def make_tables2(
        self,
        percentiles=(10, 90),
        actuals_start=pd.Timestamp("2007-01-01 00:00:00", tz="utc"),
        actuals_end=pd.Timestamp("2007-12-31 23:55:00", tz="utc"),
        scenarios_start=pd.Timestamp("2008-01-01 00:00:00", tz="utc"),
        scenarios_end=pd.Timestamp("2013-12-31 23:55:00", tz="utc"),
        source="AWS",
        **kwargs,
    ):
        """ Method retrieves data from wtk and makes actuals(DataFrame) and scenarios(dictionary) containing power conditioned (low, medium, and high) tables(DataFrame)
            and quantiles(ndarray) corresponding to input percentiles(tuple), default is (10,90).
            e.g. rts.make_tables(percentiles=(20,80))
            Timestamps for actuals and scenarios: pick dates from 2007-01-01 to 2013-12-31
            Required args:
                percentiles - (tuple)
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
        # total power veviations
        total_power_deviations_array = (
            scenarios_df["TotalPower"].values[1:]
            - scenarios_df["TotalPower"].values[:-1]
        )
        # drop last row
        scenarios_df.drop(scenarios_df.tail(1).index, inplace=True)
        # record deviations
        scenarios_df.iloc[:, :-1] = gen_deviations_array
        scenarios_df["Deviation"] = total_power_deviations_array

        # power conditioning:
        # create total power conditioning tables based on chosen percentiles
        # i.e. splits scenarios_df into three subsets (low, medium, high)
        # should be generalized
        scenarios = {}

        # get quantiles
        quantiles = np.percentile(scenarios_df["TotalPower"].values, percentiles)
        scenarios["quantiles"] = quantiles
        # print('quantiles = {}'.format(quantiles))

        # create tables

        scenarios["low"] = scenarios_df.loc[
            scenarios_df["TotalPower"] < quantiles[0]
        ].copy()
        scenarios["medium"] = scenarios_df.loc[
            (scenarios_df["TotalPower"] > quantiles[0])
            & (scenarios_df["TotalPower"] < quantiles[1])
        ].copy()
        scenarios["high"] = scenarios_df.loc[
            scenarios_df["TotalPower"] > quantiles[1]
        ].copy()

        self.scenarios = scenarios

        # new method

    def generate_wind_scenarios(
        self,
        timestamp,
        power_quantiles=[0.0, 0.1, 0.9, 1.0],
        sampling_method="monte carlo",
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

            # compute costs of each 1-period scenario
            cost_1 = scenarios_df["Deviation"].apply(
                lambda dev: np.abs(loss_of_load_cost * dev)
                if dev < 0
                else np.abs(spilled_wind_cost * dev)
            )
            # print("any neg values in cost_1? {}".format((cost_1 < 0).any()))

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

            # IS weights: f(s)/g(s), i.e. nominal/importance
            importance_weights = (1 / p_bin.size) / importance_probs.loc[sample_timestamps]

            # df of weights to return
            weights_df = pd.DataFrame(index=[timestamps[1]], columns=range(1, n_scenarios + 1))
            weights_df.loc[timestamps[1]] = dict(
                zip(range(1, n_scenarios + 1), importance_weights.values)
            )

        # initialize multi-indexed df for all scenarios to return

        iterables = [[timestamps[1]], range(1, n_scenarios + 1), timestamps[1:]]
        # iterables

        index = pd.MultiIndex.from_product(
            iterables, names=["sim_timestamp", "scenario_nr", "period_timestamp"]
        )
        # index
        multi_scenarios_df = pd.DataFrame(index=index, columns=actuals_df.columns[:-1])
        # multi_scenarios_df

        # now find wanted periods for each scenario i.e. consecutive timestamps in scenario_df
        for sample_i, sample_timestamp in enumerate(sample_timestamps):
            #print("sample_timestamp={}".format(sample_timestamp))
            # needed timestamps
            # pd.date_range(start=sample_timestamp, periods=n_periods, freq="5min")
            # deviation will be a df even with 1 period because loc is used with pd.date_range
            deviation = scenarios_df.loc[
                pd.date_range(start=sample_timestamp, periods=n_periods, freq="5min")
            ]

            # change deviation index to match actuals we adding it to
            deviation.index = timestamps[1:]

            # make scenario df for each sample_timestamp (could be done outside the loop)
            scenario_df = pd.DataFrame(
                index=timestamps[1:], columns=deviation.columns
            ).drop(["Deviation", "TotalPower"], axis=1)
            #     print("\nscenario_df:")
            #     scenario_df

            #     print("\nactual:")
            #     actuals_df.loc[timestamps[0]]

            # first take actual
            running_sum = actuals_df.loc[timestamps[0]].drop("TotalPower")
            # now keep adding deviations to the above actual
            for timestamp in timestamps[1:]:
                running_sum += deviation.loc[timestamp].drop(
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


    def generate_wind_scenarios2(
        self,
        timestamps,
        sampling_method="monte carlo",
        n_scenarios=5,
        random_seed=25,
        output_format=0,
        **kwargs,
    ):
        """Function to generate scenarios
        Required Args:
            timestamps - (itterable of pd.Timestamp) [t0,t1,...tn]
                t0 - initial/current timestamp, power conditioning will be done based on the actual power at this timestamp 
                t1 - time for wich we're making dispatch decisions, 
                t2,..,tn - required only if multiperiod scenarios are needed 
            sampling_method - (string) either "importance" or "monte carlo"
            n_scenarios - (integer) number of scenarios to draw 
                if n_scenarios == 1 and sampling_method == monte carlo, will return "deterministic case" (zero deviations)

        Returns:
            actual_df - (pd.DataFrame) actual power for each GenUID (columns) at t1 (index)
            forcast_df - (pd.DataFrame) actual power for each GenUID (columns) at t0 (index), at t1 values are the same because
                of persistence forecast
            scenarios_dict - (dictionary) of scenarios: 
                key - (integer) 1 through <n_scenarios>
                value - (pd.DataFrame) one scenario: power deviatios from persitance for each BusNum (columns) 
                    at index t0, all zeros (no deviation)
                    at index t1, actual @ t1 - actual @ t0 (power deviation from t0)
                    at index t2, actual @ t2 - actual @ t0 (power deviation from t0)
                    etc...
                
        """

        # print('time_start={}'.format(time_start))
        # print('time_end={}'.format(time_end))

        # these were made with make_tables method
        actuals_df = self.actuals
        quantiles = self.scenarios.get("quantiles")
        scenarios_low_power_df = self.scenarios.get("low")
        scenarios_med_power_df = self.scenarios.get("medium")
        scenarios_high_power_df = self.scenarios.get("high")
        # tables list
        tables = [
            scenarios_low_power_df,
            scenarios_med_power_df,
            scenarios_high_power_df,
        ]
        # just concat them all (then multiperiod is much simpler)
        all_df = pd.concat(tables)
        all_df.sort_index(inplace=True)

        # basic checks: still needs work for multiperiod
        for timestamp in timestamps:
            # type
            if type(timestamp) != pd._libs.tslibs.timestamps.Timestamp:
                print("timestamps must be pandas.Timestamp type")
                return

        # t0 and t1 must be in actuals.index
        if not (timestamps[0] in actuals_df.index) or not (
            timestamps[1] in actuals_df.index
        ):
            print(
                "timestamps[0] and timestamps[1] must be between {} and {}".format(
                    actuals_df.index.min(), actuals_df.index.max()
                )
            )
            return

        # check if t0 < t1
        if not (timestamps[0] < timestamps[1]):
            print("timestamps[0] must be < timestamps[1]")
            return

        # actual power at t1 (same thing whether it is single- or multi-period)
        actual_df = actuals_df.loc[
            timestamps[1] : timestamps[1]
        ].copy()  # loc[t:t] returns frame, loc[t] returns series

        # actual power at t0, same values at index t1 (persistance forecast)
        forecast_df = pd.DataFrame(index=timestamps[:2], columns=actual_df.columns)
        forecast_df.loc[timestamps[0]] = actuals_df.loc[timestamps[0]].copy()
        forecast_df.loc[timestamps[1]] = actuals_df.loc[
            timestamps[0]
        ].copy()  # persistence

        # deterministic case: if n_scenarios == 1 and sampling_method == `monte carlo`
        if n_scenarios == 1 and sampling_method == "monte carlo":
            # initialize scenario_df (same index as forcast_df)
            scenario_df = forecast_df.copy()
            # first row of zeros, because Devon says so
            scenario_df.loc[timestamps[0]] = 0.0
            # second row of scenario_df is also 0, because it is deterministic case
            scenario_df.loc[timestamps[1]] = 0.0
            # drop TotalPower column
            scenario_df.drop("TotalPower", axis=1, inplace=True)

            scenarios_dict = {}
            scenarios_dict[1] = scenario_df

            # drop Total column from actual_df and forecast_df
            forecast_df.drop("TotalPower", axis=1, inplace=True)
            actual_df.drop("TotalPower", axis=1, inplace=True)

            return scenarios_dict, actual_df, forecast_df

        # power conditioning on total power at t0
        actual_total_power = actuals_df.loc[timestamps[0]]["TotalPower"]
        # print('Actual total power = {}'.format(actual_total_power))

        if actual_total_power < quantiles[0]:
            # draw scenarions from low power table
            # print('using low power table (total power < {:.2f})'.format(quantiles[0]))
            scenarios_df = scenarios_low_power_df.copy()
        elif actual_total_power > quantiles[1]:
            # draw scenarions from high power table
            # print('using high power table (total power > {:.2f})'.format(quantiles[1]))
            scenarios_df = scenarios_high_power_df.copy()
        else:
            scenarios_df = scenarios_med_power_df.copy()
            # draw scenarions from medium power table
        # print('using medium power table ({:.2f} < total power < {:.2f})'.format(quantiles[0],quantiles[1]))

        # add weights column (sampling probabilities) depending on the sampling method
        if sampling_method == "importance":
            # temporary solution, cost function should come in as one of the variables
            # if importance sampling (using low fidelity loss function)
            # parameters of linear loss function
            loss_of_load_cost = 1000 / 3.0
            spilled_wind_cost = 0.001
            scenarios_df["Weight"] = scenarios_df.apply(
                lambda row: row["Deviation"] * (-loss_of_load_cost)
                if row["Deviation"] < 0
                else row["Deviation"] * spilled_wind_cost,
                axis=1,
            )
            # some 'duplicate axis' error is showing when doing the bolow two lines... so lets do slow way as above
            # this should be prebuilt anyway
            # apply linear loss function to deviation column to get weight column
            # scenarios_df.loc[scenarios_df['Deviation']<0,'Weight'] = scenarios_df['Deviation']*(-loss_of_load_cost)
            # scenarios_df.loc[scenarios_df['Deviation']>0,'Weight'] = scenarios_df['Deviation']*(spilled_wind_cost)

            # normalize
            scenarios_df["Weight"] = (
                scenarios_df["Weight"] / scenarios_df["Weight"].sum()
            )

        elif sampling_method == "monte carlo":
            # if monte carlo method, all weights are equal
            scenarios_df["Weight"] = 1.0 / len(scenarios_df)

        # now draw random sample using weight column (the larger the weight the more likely it is to draw that sample)
        # draw n_scenarios, one by one, accept only if tn is before t_final (which is 2013-12-31 23:50:00, but it could different)
        # max_index_in_tables = max(scenarios_low_power_df.index.max(),scenarios_med_power_df.index.max(),scenarios_high_power_df.index.max())
        # same as a bove in a more pythonic way
        # max_index_in_tables = max([table.index.max() for table in tables])
        # max_index_in_tables = all_df.index.max()

        # time gaps betweet t1 and ti, i=2,3,...n
        # for single-period this will be empty list
        time_gaps_bw_t1_ti = [timestamp - timestamps[1] for timestamp in timestamps[2:]]
        scenarios_dict = {}
        key = 1
        while len(scenarios_dict) < n_scenarios:
            # print('key={}'.format(key))
            # print('len(scenarios_dict)={}'.format(len(scenarios_dict)))
            # while we haven't filled up scenario dict, keep sampling
            scenario_sample_df = scenarios_df.sample(
                n=1, weights="Weight", random_state=random_seed + key
            )

            sample_timestamp = scenario_sample_df.index[0]
            # print("sample_timestamp={}".format(sample_timestamp))
            # print('sample_timestamp={}'.format(sample_timestamp))
            # check if all indices (every 5-min) from sample_timestamp to sample_timestamp + time_gap_bw_t1_tn are in the tables (if not start over)
            # this check only needed only if multi-period scenarios are required i.e. len(timestamps)>2
            if time_gaps_bw_t1_ti:
                needed_timestamps = pd.date_range(
                    sample_timestamp,
                    sample_timestamp + time_gaps_bw_t1_ti[-1],
                    freq="5min",
                )
            else:
                needed_timestamps = []

            for timestamp in needed_timestamps:
                if not timestamp in all_df.index:
                    continue

            needed_timestamps_df = all_df.loc[needed_timestamps]

            # drop Deviation and Weight column, no longer needed (they are for conditioning and sampling)
            scenario_sample_df.drop(["Deviation", "Weight"], axis=1, inplace=True)

            # make scenario df
            # initialize scenario_df (index as given timestamps)
            scenario_df = pd.DataFrame(
                index=timestamps, columns=scenario_sample_df.columns
            )
            # deviations at index t0 are all zeros
            scenario_df.loc[timestamps[0]] = 0.0
            # deviations at index t1 are the scenarios_sample_df's only entry
            scenario_df.loc[timestamps[1]] = scenario_sample_df.iloc[0].copy()
            # all subsequent entries at t2,...,tn are found according to the time distance from t1 (running sum)
            for i, timestamp in enumerate(timestamps[2:]):

                # print("timestamp={}".format(timestamp))
                # timedelta from t1
                # once we stepped all the way to timestamp, record moving sum to the scenario_df

                scenario_df.loc[timestamp] = needed_timestamps_df.loc[
                    sample_timestamp : sample_timestamp + time_gaps_bw_t1_ti[i]
                ].sum()

            # drop TotalPower column
            # print('scenario_df columns:')
            # print(scenario_df.columns)
            scenario_df.drop("TotalPower", axis=1, inplace=True)

            ##### fix "over" and "under" capacity problem
            # "over" = if scenario power (forecast + scenario) goes over GenMWMax capacity
            # "under" = if scenario power (forecast + scenario) goes under 0 i.e. negative generation

            # for GenMWMax info we take wind generators and reindex by GenUID
            wind_generators_df = self.wind_generators.set_index(
                "GenUID"
            )  # this returns a reindexed copy
            for timestamp in scenario_df.index[1:]:

                # "over"
                # if element of over is negative, we need to add it to current and consecutive timestamps
                # (if positive don't do anything. we accomplish this by zeroing positive elements of over)
                over = wind_generators_df["GenMWMax"] - (
                    forecast_df.iloc[0].drop("TotalPower") + scenario_df.loc[timestamp]
                )
                # if there are any negetive numbers in over, add the to the tail of scenario_df
                if (over < 0).any():
                    over[over > 0.0] = 0.0
                    scenario_df.loc[timestamp:] = scenario_df.loc[timestamp:] + over

                # "under"
                # if element of under is negative, we need to subract it from current and consecutive timestamps
                # (if positive don't do anything. we accomplish this by zeroing positive elements of under)
                under = (
                    forecast_df.iloc[0].drop("TotalPower") + scenario_df.loc[timestamp]
                )
                if (under < 0).any():
                    under[under > 0.0] = 0.0
                    scenario_df.loc[timestamp:] = scenario_df.loc[timestamp:] - under

            # add scenario_df to scenarios_dict
            scenarios_dict[key] = scenario_df
            key += 1

        # drop Total column from actual_df and forecast_df
        forecast_df.drop("TotalPower", axis=1, inplace=True)
        actual_df.drop("TotalPower", axis=1, inplace=True)

        if output_format == 0:
            return scenarios_dict, actual_df, forecast_df
        # make multi indexed df out of the scenarios_dict

        elif output_format == 1:
            actual_s = actual_df.iloc[0]
            forecast_s = forecast_df.iloc[0]
            iterables = [
                [timestamps[1]],
                list(scenarios_dict.keys()),
                scenarios_dict[1].index[1:].to_list(),
            ]
            index = pd.MultiIndex.from_product(
                iterables, names=["sim_timestamp", "scenario_nr", "period_timestamp"]
            )
            multi_scenarios_df = pd.DataFrame(
                index=index, columns=scenarios_dict[1].columns
            )

            # stick data to df from dict
            for i, df in scenarios_dict.items():
                # print("scenario: {}".format(i))
                period_timestamps = df.index.to_list()
                for period_timestamp in period_timestamps[
                    1:
                ]:  # don't take the first one
                    # print("period timestamp: {}".format(period_timestamp))
                    multi_scenarios_df.loc[timestamps[1], i, period_timestamp] = (
                        df.loc[period_timestamp] + forecast_s
                    )

            # return total wind power (as opposed to deviation)
            # multi_scenarios_df=multi_scenarios_df+forecast_s

            return multi_scenarios_df, actual_s
