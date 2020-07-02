"""
Main script to generate scenarios
"""

# System
import os
import sys
import argparse
import logging

# Externals
import yaml
import numpy as np
import pandas as pd


# Locals
from powerscenarios.parser import Parser
from powerscenarios.grid import Grid


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("generate_scenarios.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="config.yaml")
    # add_arg('-d', '--distributed', action='store_true')
    add_arg("-v", "--verbose", action="store_true")
    # parameters which override the YAML file, if needed
    #
    return parser.parse_args()


def config_logging(verbose):
    # log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    # logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    # define file handler and set formatter
    file_handler = logging.FileHandler("logfile.log")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    return logger


def load_config(args):
    # Read base config from yaml file
    config_file = args.config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # can override with command line arguments here if needed

    return config


def main():
    """ Main function """
    # Initialization
    args = parse_args()

    # Load configuration
    config = load_config(args)
    output_dir = os.path.expandvars(config["output_dir"])

    os.makedirs(output_dir, exist_ok=True)

    # Loggging
    logger = config_logging(verbose=args.verbose)

    # load grid data
    # path to .aux file (TAMU grids) obtained from e.g.
    # https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg200/
    data_dir = os.path.expandvars(config["data_dir"])
    grid_name = config["grid"]["name"]
    aux_file_name = data_dir + grid_name + "/" + grid_name + ".aux"
    # parse original .aux file and return dataframes for buses, generators, and wind generators
    # here, we need .aux files because those are the only ones with Latitute/Longitude information
    parser = Parser()
    bus_df, gen_df, wind_gen_df = parser.parse_tamu_aux(aux_file_name)

    # to instantiate a grid we need: name, bus, generator, and wind generator dataframes from Parser
    # really, we only wind generators with lat/long, might change in the future
    grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)
    logger.info(grid.info())
    # print(grid.info())

    # retrieve wind sites (wind_sites are initially set to empty df )
    grid.retrieve_wind_sites()


    # sampling tables
    grid.make_tables(
        actuals_start=pd.Timestamp(config["tables"]["actuals_start"], tz="utc"),
        actuals_end=pd.Timestamp(config["tables"]["actuals_end"], tz="utc"),
        scenarios_start=pd.Timestamp(config["tables"]["scenarios_start"], tz="utc"),
        scenarios_end=pd.Timestamp(config["tables"]["scenarios_end"], tz="utc"),
        source="AWS",
    )
    



    # timespan for wanted scenarios
    # time period for which to generate scenarios
    scenario_start = pd.Timestamp(config["scenario"]["start"], tz="utc")
    scenario_end = pd.Timestamp(config["scenario"]["end"], tz="utc")
    # for actuals, make year you want ( e.g. TAMU 2000 load is for 2017, so match that )
    grid.actuals.index = grid.actuals.index.map(lambda t: t.replace(year=scenario_start.year))

    sim_timestamps = pd.date_range(start=scenario_start, end=scenario_end, freq="5min")
    # other parameters
    sampling_method = config["scenario"]["sampling_method"]
    #sampling_method="importance"
    n_scenarios = config["scenario"]["n_scenarios"]
    n_periods = config["scenario"]["n_periods"] 



    all_scenarios_df = pd.DataFrame()
    all_weights_df = pd.DataFrame()

    for sim_timestamp in sim_timestamps:
        print("sim_timestamp = {}".format(sim_timestamp))
        random_seed = np.random.randint(2 ** 31 - 1)
        #random_seed = 594081473
        print("random_seed = {}".format(random_seed))
        scenarios_df, weights_df = grid.generate_wind_scenarios(
            sim_timestamp,
            power_quantiles=[0.0, 0.1, 0.9, 1.0],
            sampling_method=sampling_method,
            n_scenarios=n_scenarios,
            n_periods=n_periods,
            # random_seed=6,
            random_seed=random_seed,
            output_format=0,
        )
        all_scenarios_df=pd.concat([all_scenarios_df,scenarios_df])
        all_weights_df=pd.concat([all_weights_df,weights_df])
            


  ################# CLEAN UP AND SAVE 

    selected_actuals_df = grid.actuals.loc[sim_timestamps]

    # rename columns to match Maack's convention
    selected_actuals_df.index.rename("DateTime", inplace=True)
    selected_actuals_df.drop("TotalPower",axis=1, inplace=True)

    # optional column renaming
    old_names = selected_actuals_df.columns.values
    new_names = [name.replace("Wind", "WIND") for name in old_names]
    selected_actuals_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    # drop tz, then julia's CSV can parse dates 
    selected_actuals_df.index = selected_actuals_df.index.map(lambda t: t.replace(tzinfo=None))


    filename = output_dir + 'actuals_'+ grid_name+ '.csv'
    selected_actuals_df.to_csv(filename)

    
    # optional column renaming
    old_names = all_scenarios_df.columns.values
    new_names = [name.replace("Wind", "WIND") for name in old_names]
    all_scenarios_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    # drop tz from scenarios as well, then julia's CSV can parse dates 
    all_scenarios_df.index=all_scenarios_df.index.map(
        lambda t: (t[0].replace(tzinfo=None), t[1], t[2].replace(tzinfo=None))
    )

    filename = output_dir + 'scenarios_'+ grid_name+ '.csv'
    all_scenarios_df.to_csv(filename)
 
    # save weights
    filename = output_dir + 'weights_'+ grid_name+ '.csv'
    all_weights_df.to_csv(filename)





















if __name__ == "__main__":
    main()
