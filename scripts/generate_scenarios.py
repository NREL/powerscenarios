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


def config_logging(verbose,output_dir):
    # log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    # logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    # define file handler and set formatter
    file_handler = logging.FileHandler(os.path.join(output_dir,"logfile.log"))
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
    output_dir = os.path.expandvars(config["output"]["dir"])

    os.makedirs(output_dir, exist_ok=True)

    # Loggging
    logger = config_logging(verbose=args.verbose,output_dir=output_dir)

    data_dir = os.path.expandvars(config["data_dir"])
    grid_name = config["grid"]["name"]

    # load TAMU grid data
    # path to .aux file (TAMU grids) obtained from e.g.
    # https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg200/
    if grid_name[:7]  == 'ACTIVSg':
        #aux_file_name = data_dir + grid_name + "/" + grid_name + ".aux"
        aux_file_name = os.path.join(data_dir, grid_name, grid_name + ".aux")
        # parse original .aux file and return dataframes for buses, generators, and wind generators
        # here, we need .aux files because those are the only ones with Latitute/Longitude information
        parser = Parser()
        bus_df, gen_df, wind_gen_df = parser.parse_tamu_aux(aux_file_name)

    elif grid_name == 'RTS':
        bus_csv_filename = os.path.join(data_dir, grid_name, "bus.csv")
        gen_csv_filename = os.path.join(data_dir, grid_name, "gen.csv")

        parser = Parser()
        # if solar2wind, will replace all solar with wind
        bus_df, gen_df, wind_gen_df = parser.parse_rts_csvs(
            bus_csv_filename, gen_csv_filename, solar2wind=config["RTS_solar2wind"]
        )




    # to instantiate a grid we need: name, bus, generator, and wind generator dataframes from Parser
    # really, we only wind generators with lat/long, might change in the future
    grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)
    logger.info(grid.info())
    # print(grid.info())

    if config["wind_penetration"]["change"]==True:
        logger.info("changing wind penetration")
        grid.change_wind_penetration(config["wind_penetration"]["new_value"])
        logger.info(grid.info())


    logger.info("retrieving wind sites")
    # retrieve wind sites (wind_sites are initially set to empty df )
    grid.retrieve_wind_sites(method="simple proximity")


    logger.info("making tables")
    # sampling tables
    grid.make_tables(
        actuals_start=pd.Timestamp(config["tables"]["actuals_start"], tz="utc"),
        actuals_end=pd.Timestamp(config["tables"]["actuals_end"], tz="utc"),
        scenarios_start=pd.Timestamp(config["tables"]["scenarios_start"], tz="utc"),
        scenarios_end=pd.Timestamp(config["tables"]["scenarios_end"], tz="utc"),
    )
    



    logger.info("generating scenarios")
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

    ########################################################

    all_weights_df = pd.DataFrame(index=sim_timestamps,columns=range(1,n_scenarios+1))

    # create multiindex df for all generated scenarios
    # three arrays for multiindex:
    a1 = [x for x in sim_timestamps for k in range(n_scenarios * n_periods)]
    a2 = [x for x in range(1, n_scenarios + 1) for k in range(n_periods)] * len(
        sim_timestamps
    )
    a3 = [
        t + pd.Timedelta("5min") * k
        for t in sim_timestamps
        for k in list(range(n_periods)) * n_scenarios
    ]

    index = pd.MultiIndex.from_arrays([a1,a2,a3],names=['sim_timestamp','scenario_nr','period_timestamp'])
    all_scenarios_df = pd.DataFrame(index=index,columns=grid.wind_generators["GenUID"].values)



    for sim_timestamp in sim_timestamps:
        logger.info("sim_timestamp = {}".format(sim_timestamp))
        random_seed = np.random.randint(2 ** 31 - 1)
        #random_seed = 594081473
        #print("random_seed = {}".format(random_seed))
        scenarios_df, weights_df = grid.generate_wind_scenarios(
            sim_timestamp,
            power_quantiles=[0.0, 0.1, 0.9, 1.0],
            sampling_method=sampling_method,
            n_scenarios=n_scenarios,
            n_periods=n_periods,
            #random_seed=6,
            random_seed=random_seed,
            output_format=0,
        )
        #all_scenarios_df=pd.concat([all_scenarios_df,scenarios_df])
        all_scenarios_df.loc[sim_timestamp]=scenarios_df

        #all_weights_df=pd.concat([all_weights_df,weights_df])
        all_weights_df.loc[sim_timestamp]=weights_df.loc[sim_timestamp]




    all_actuals_df=grid.actuals.loc[sim_timestamps].drop("TotalPower", axis=1).copy()

  ################## CLEAN

    if config["output"]["df_format"] == "julia maack":
        # rename columns to match Maack's convention
        all_actuals_df.index.rename("DateTime", inplace=True)

        # optional column renaming
        old_names = all_actuals_df.columns.values
        new_names = [name.replace("Wind", "WIND") for name in old_names]
        all_actuals_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        # optional drop tz, then julia's CSV can parse dates 
        all_actuals_df.index = all_actuals_df.index.map(lambda t: t.replace(tzinfo=None))      


        # optional column renaming
        old_names = all_scenarios_df.columns.values
        new_names = [name.replace("Wind", "WIND") for name in old_names]
        all_scenarios_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        # optional drop tz from scenarios as well, then julia's CSV can parse dates 
        all_scenarios_df.index=all_scenarios_df.index.map(
            lambda t: (t[0].replace(tzinfo=None), t[1], t[2].replace(tzinfo=None))
        )

  ################# SAVE 
    if config["output"]["file_format"] == 'csv':
        
        # save actuals
        filename = os.path.join(output_dir,'actuals_'+ grid_name+ '.csv')
        logger.info("\nsaving all_actuals_df to {}".format(filename))
        all_actuals_df.to_csv(filename)

        # save scenarios 
        filename = os.path.join(output_dir,'scenarios_'+ grid_name+ '.csv')
        logger.info("\nsaving all_scenarios_df to {}".format(filename))
        all_scenarios_df.to_csv(filename)
     
        # save weights
        #filename = output_dir + 'weights_'+ grid_name+ '.csv'
        filename = os.path.join(output_dir,'weights_'+ grid_name+ '.csv')
        logger.info("\nsaving all_weights_df to {}".format(filename))
        all_weights_df.to_csv(filename)


    elif config["output"]["file_format"] == 'aux':

        # choose delimiter: could be implemented as an argument to a function
        #delimiter = " "
        delimiter = "\t"
        ###################################### save scenarios
        for i in range(len(all_scenarios_df)):

            s = all_scenarios_df.iloc[i]

            # create filename from multiindex, option 1: <simulation timestamp>_<scenario number>_<period timestamp>
            # filename = str(s.name[0]).replace(' ','-')+'_'+str(s.name[1])+'_'+str(s.name[2]).replace(' ','-')

            # create filename from multiindex, option 2: <simulation timestamp>_<scenario number>_<period number>
            filename = (
                str(s.name[0]).replace(" ", "-")
                + "_S"
                + str(s.name[1])
                + "_P"
                + str(((s.name[2] - s.name[0]).seconds // 60) // 5 + 1)
                + ".aux"
            )

            filename = os.path.join(output_dir, filename) 


            with open(filename, "w") as f:
                # .aux header
                _ = f.write("DATA (Gen, [BusNum,GenID,GenMW])\n")
                _ = f.write("{\n")
                # each series entry is one line
                for k in range(len(s)):
                    _ = f.write(
                        delimiter
                        + s.index[k].split("_")[0]
                        + delimiter
                        + '"'
                        + s.index[k].split("_")[2]
                        + '"'
                        + delimiter
                        + str(s[k])
                        + delimiter
                        + '"Closed"'
                        + "\n"
                    )
                # .aux EOF
                _ = f.write("}\n")


        ######################################## save actuals

        for i in range(len(all_actuals_df)):

            s = all_actuals_df.iloc[i]

            # create filename from multiindex, option 2: <simulation timestamp>_<scenario number>_<period number>
            filename = str(s.name).replace(" ", "-") + "_A" + ".aux"

            filename = os.path.join(output_dir, filename) 

            with open(filename, "w") as f:
                # .aux header
                _ = f.write("DATA (Gen, [BusNum,GenID,GenMW])\n")
                _ = f.write("{\n")
                # each series entry is one line
                for k in range(len(s)):
                    _ = f.write(
                        delimiter
                        + s.index[k].split("_")[0]
                        + delimiter
                        + '"'
                        + s.index[k].split("_")[2]
                        + '"'
                        + delimiter
                        + str(s[k])
                        + delimiter
                        + '"Closed"'
                        + "\n"
                    )
                # .aux EOF
                _ = f.write("}\n")

















if __name__ == "__main__":
    main()
