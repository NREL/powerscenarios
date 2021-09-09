from mpi4py import MPI
import sys
import glob
import pandas as pd
import time


import pandas as pd
import numpy as np
import time
import os

import sys
import argparse
import logging
import yaml


######
# # set PYWTK_CACHE_DIR to locate WIND Toolkit data 
# # will download from AWS as needed
# os.environ["PYWTK_CACHE_DIR"] = os.path.join(os.environ["HOME"], "pywtk-data")

from powerscenarios.parser import Parser

# from powerscenarios.grid import Grid
from powerscenarios.grid_copy2 import Grid

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("buildout.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="config_simple_mpi.yml")
    # add_arg('-d', '--distributed', action='store_true')
    # parameters which override the YAML file, if needed
    add_arg("-v", "--verbose", action="store_true")
    add_arg("-o", "--output_dir", dest="output_dir", type=str, help="output directory")
    # add_arg("-m", "--method", dest="method", type=str, help="buildout method: voronoi or radius", )
    # add_arg("-r", "--max_radius", dest="max_radius", type=float, help="maximum radius around the bus; required for 'radius' method", )
    # add_arg(
    #     "-p",
    #     "--penetration_level",
    #     dest="penetration_level",
    #     type=float,
    #     help="penetration level (e.g. 0.2 for 20%)",
    # )
    return parser.parse_args()


def config_logging(verbose, output_dir):
    # log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    # logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    # define file handler and set formatter
    file_handler = logging.FileHandler(os.path.join(output_dir, "log_checkmark_mpi.log"))
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)

    ### console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def load_config(args):
    ## Read base config from file (yaml or json)
    config_file = args.config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = json.load(f)
        # can override with command line arguments here if needed
        if args.output_dir:
            config["output_dir"] = args.output_dir
        # if args.penetration_level:
        #     config["penetration_level"] = args.penetration_level
        # if args.method:
        #     config["method"] = args.method
        # if args.max_radius:
        #     config["max_radius"] = args.max_radius

    return config



def checkmark_cost(total_deviation):
    loss_of_load_cost = 10000 / 12.0
    spilled_wind_cost = 0.001
    if total_deviation < 0:
        cost = loss_of_load_cost * abs(total_deviation)
    else:
        cost = spilled_wind_cost * abs(total_deviation)
    return cost




comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()




def main():
    """ Main function """
    # Initialization
    args = parse_args()


    # Load configuration
    config = load_config(args)

    #data_dir =     
    output_dir = os.path.expandvars(config["output"]["dir"])
    os.makedirs(output_dir, exist_ok=True)

    # # if method ir radius and no max_radius given raise error
    # if (config["method"] == "radius") and (not config["max_radius"]):
    #     raise ValueError("Option 'max_radius' not given.")

     
    #output_dir = os.path.expandvars(config["output_dir"])

    # # input dir needs to contain sources_with_network templates
    # input_dir = os.path.expandvars(config["input_dir"])

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # os.makedirs(output_dir, exist_ok=True)                                                                            

    

    # Loggging
    logger = config_logging(verbose=args.verbose, output_dir=output_dir)



    if rank == 0:
        #print("\n\n\n#########################################################")
        #print("\n\n\n#########################################################")

        #grid_name = "ACTIVSg200"  # TAMU 200 bus casea
        grid_name = config["grid"]["name"]

        # grid_name = "ACTIVSg2000"  # TAMU 2000 bus case
        # grid_name = "ACTIVSg10k"  # TAMU 10000 bus case

        # path to .aux file (TAMU grids) obtained from e.g.
        # https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg200/
        # data_dir = "../../data/grid-data/"

        data_dir = os.path.expandvars(config["input"]["grid_dir"])

        #data_dir = "../data/grid-data/"

        # aux_file_name = data_dir + grid_name + "/" + grid_name + ".aux"
        aux_file_name = os.path.join(data_dir, grid_name, grid_name + ".aux")

        # parse original .aux file and return dataframes for buses, generators, and wind generators
        # here, we need .aux files because those are the only ones with Latitute/Longitude information
        parser = Parser()
        bus_df, gen_df, wind_gen_df = parser.parse_tamu_aux(aux_file_name)

        # to instantiate a grid we need: name, bus, generator, and wind generator dataframes from Parser
        # really, we only wind generators, will change in the future
        grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)
        grid
        #print(grid.info())
        logger.debug(grid.info())

        ## if read_tables then use h5 wind_sites/actuals/scenarios tables
        if config["input"]["read_tables"]:
            tables_dir = os.path.expandvars(config["input"]["tables_dir"])


            # read instead of retrieve_wind_sites
            filename = "{}_wind_sites_df.h5".format(grid_name)
            grid.wind_sites = pd.read_hdf(os.path.join(tables_dir,filename))
            #print("\n wind sites")
            #print(grid.wind_sites.head())
            logger.debug("\n wind sites")
            logger.debug(grid.wind_sites.head())

            # read instead of make_tables
            filename = "{}_actuals_df.h5".format(grid_name)
            grid.actuals = pd.read_hdf(os.path.join(tables_dir,filename))
            filename = "{}_scenarios_df.h5".format(grid_name)
            grid.scenarios = pd.read_hdf(os.path.join(tables_dir,filename))

        # for actuals, make year you want
        grid.actuals.index = grid.actuals.index.map(lambda t: t.replace(year=2020))
        # see what you got
        #print("\nactuals_df:")
        #print(grid.actuals.head())
        #print("\nscenarios_df:")
        #print(grid.scenarios.head())
        logger.debug("\n\nactuals_df:")
        logger.debug(grid.actuals.head())
        logger.debug("\n\nscenarios_df:")
        logger.debug(grid.scenarios.head())

        ## for TESTING just take a smaller scenario df
        ## first 10 (all positive deviations)
        #s_df = grid.scenarios.head(100).copy()
        #s_df = grid.scenarios.head(100000).copy()
        s_df = grid.scenarios.copy()
        ## ten scenarios with first 5 positive and last 5 negative deviations

        # s_df = grid.scenarios.loc[
        #     pd.Timestamp("2008-01-01 03:00:00+00:00") : pd.Timestamp(
        #         "2008-01-01 03:45:00+00:00"
        #     )
        # ].copy()


        ## drop Total power and Deviation (but remember thsese, will need later for power conditioning)
        s_df_Deviation = s_df["Deviation"].copy()
        s_df_TotalPower = s_df["TotalPower"].copy()
        s_df.drop(["Deviation", "TotalPower"], axis=1, inplace=True)

        ## add Cost placer
        s_df["Cost"] = 0
        s_df.reset_index(inplace=True)
        s_df["IssueTime"] = s_df["IssueTime"].astype(str)

        logger.debug("\n\ns_df:")
        logger.debug(s_df)

        #### remember index and columns of scenarios df before turning it into the list
        s_idx = s_df.index
        s_col = s_df.columns

        s_list = s_df.values.tolist()
        logger.debug("\n\ns_list:")
        logger.debug(s_list)

        # scatter a list

        t0 = time.time()

        #print(" i'm rank {}:".format(rank))
        logger.debug("\n processing {} files on {} cores".format(len(s_list), size))

        # don't know how to implement this check with ranks, sys.exit() seems to work, but maybe I need to do comm.Abort() ?
        # check if number of ranks <= number of files
        if size > len(s_list):
            print(
                "number of scenarios {} < number of ranks {}, abborting...".format(
                    len(s_list), size
                )
            )
            sys.exit()

        # split them into chunks (number of chunks = number of ranks)
        chunk_size = len(s_list) // size

        remainder_size = len(s_list) % size

        s_list_chunks = [
            s_list[i : i + chunk_size] for i in range(0, size * chunk_size, chunk_size)
        ]
        # distribute remainder to chunks
        for i in range(-remainder_size, 0):
            s_list_chunks[i].append(s_list[i])

        logger.debug("\nscattering data:")
        logger.debug(s_list_chunks)

    else:
        s_list_chunks = None

    ##print("\n")
    # print(s_list_chunks)
    # s_list_chunks


    s_list_chunks = comm.scatter(s_list_chunks, root=0)
    # print(
    #     "\n\n\nrank {} has {} scenarios = {}".format(
    #         rank, len(s_list_chunks), s_list_chunks
    #     )
    # )
    # print('\n\n\nrank {} has scenarios = {}'.format(rank,s_list_chunks))


    # now change scattered data
    for scenario in s_list_chunks:
        total_deviation = sum(scenario[1:-1])
        # print("\n\nrank {}: total_deviation for scenario {} is {}".format(rank, scenario[0], total_deviation ))
        cost = checkmark_cost(sum(scenario[1:-1]))
        logger.debug(
            "\n\nrank {}: total_deviation for scenario {} is {} and its cost is {}".format(
                rank, scenario[0], total_deviation, cost
            )
        )
        # recort this cost as last element of the list
        scenario[-1] = cost
        # max_value, index = find_max(file_name[2])
        # file_name[0] = max_value
        # file_name[1] = index


    # gather s_list_chunks back to one list
    new_s_list_chunks = comm.gather(s_list_chunks, root=0)

    if rank == 0:
        logger.debug(
            "\n\nrank {}: gathered new_s_list_chunks is: {}".format(rank, new_s_list_chunks)
        )
        new_s_list = [scenario for chunk in new_s_list_chunks for scenario in chunk]

        ## now assemble original scenario_df
        new_s_df = pd.DataFrame(index=s_idx, columns=s_col, data=new_s_list)

        ## new_s_df will not be sorted by IssueTime
        new_s_df.set_index("IssueTime", inplace=True)
        ## make index datetime
        new_s_df.index = pd.to_datetime(new_s_df.index)
        new_s_df.sort_index(inplace=True)

        ## add back Deviation and TotalPower
        new_s_df["Deviation"] = s_df_Deviation
        new_s_df["TotalPower"] = s_df_TotalPower

        #### make grid.scenarios to be the new dataframe wirh mpi computed costs
        #### add cost_1 to grid.scenarios
        grid.scenarios = new_s_df

        



        # ##### for debugging purposes record what you got
        # logger.info("\n\n rank {} has new_s_df:".format(rank))
        # logger.info(new_s_df)

        # logger.info("Priced {} scenarios".format(len(new_s_df) ) )
        # t1 = time.time()
        # logger.info("Ellapsed time: {} s".format(t1-t0))
        
        # ## save output to hdf        
        # filename = "{}_new_s_df.h5".format(grid_name)
        # new_s_df.to_hdf(os.path.join(output_dir,filename),'MW',  mode='w')
        # logger.info("saving output to {}".format(os.path.join(output_dir,filename)))

        # ## save output to csv        
        # filename = "{}_new_s_df.csv".format(grid_name)
        # new_s_df.to_csv(os.path.join(output_dir,filename))
        # logger.info("saving output to {}".format(os.path.join(output_dir,filename)))

        ####### generate scenarios
        # other parameters
        #sim_timestamp = pd.Timestamp(config["scenario"]["start"])
        sim_timestamp = pd.Timestamp("2020-08-01 00:15:00+0000", tz="UTC")

        logger.info("\nsim_timestamp = {}".format(sim_timestamp))

        sampling_method = config["scenario"]["sampling_method"]
        fidelity = config["scenario"]["fidelity"]
        n_scenarios = config["scenario"]["n_scenarios"] 
        n_periods =  config["scenario"]["n_periods"]


        # sampling_method = "importance" 
        # fidelity = "checkmark"
        # n_scenarios = 3 
        # n_periods = 1



        # random_seed = np.random.randint(2 ** 31 - 1)
        random_seed = 25


        scenarios_df, weights_df, p_bin, cost_n = grid.generate_wind_scenarios2(
            sim_timestamp,
            power_quantiles=[0.0, 0.1, 0.9, 1.0],
            sampling_method=sampling_method,
            fidelity=fidelity,
            n_scenarios=n_scenarios,
            n_periods=n_periods,
            # random_seed=6,
            random_seed=random_seed,
            output_format=0,
        )


        ##### save generated dataframes
        save_dir = config["output"]["dir"]
        actuals_df = (
           grid.actuals.loc[sim_timestamp:sim_timestamp].drop("TotalPower", axis=1).copy()
        )

        if config["output"]["df_format"] == "Shri":

            # add t0 actual
            actuals_df.loc[sim_timestamp - pd.Timedelta("5min")] = grid.actuals.loc[
                    sim_timestamp - pd.Timedelta("5min")
                ]
            actuals_df.sort_index(inplace=True)
            ## save actuals_df
            filename = "actuals_{}.csv".format(grid_name)
            actuals_df.to_csv(os.path.join(save_dir, filename))

            df = scenarios_df.loc[sim_timestamp]
            # drop period_timestamp multi-index level
            mindex = df.index
            df.index = mindex.droplevel(1)

            ## add weights column
            df["weight"] = weights_df.loc[sim_timestamp]

            ## save scenarios with weights
            filename = "{}_scenarios_{}.csv".format(n_scenarios, grid_name)
            df.to_csv(os.path.join(save_dir, filename))
       
        elif config["output"]["df_format"] == "original":

            if config["output"]["file_format"] == "csv":

                ## add t0 actual
                actuals_df.loc[sim_timestamp - pd.Timedelta("5min")] = grid.actuals.loc[
                        sim_timestamp - pd.Timedelta("5min")
                    ]
                actuals_df.sort_index(inplace=True)

                ## save actuals_df
                filename = "orig_actuals_{}.csv".format(grid_name)
                actuals_df.to_csv(os.path.join(save_dir, filename))

                ## save scenarios
                filename = "{}orig_scenarios_{}.csv".format(n_scenarios, grid_name)
                scenarios_df.to_csv(os.path.join(save_dir, filename))

                ## save weights
                filename = "{}orig_weights_{}.csv".format(n_scenarios, grid_name)
                weights_df.to_csv(os.path.join(save_dir, filename))



            # #### todo: problem with multiindex when saving to h5
            # elif config["output"]["file_format"] == "h5":

            #     ## add t0 actual
            #     actuals_df.loc[sim_timestamp - pd.Timedelta("5min")] = grid.actuals.loc[
            #             sim_timestamp - pd.Timedelta("5min")
            #         ]
            #     actuals_df.sort_index(inplace=True)

            #     ## save actuals_df
            #     filename = "orig_actuals_{}.h5".format(grid_name)
            #     actuals_df.to_hdf(os.path.join(save_dir, filename),"MW", mode="w")

            #     ## save scenarios
            #     filename = "{}orig_scenarios_{}.h5".format(n_scenarios, grid_name)
            #     scenarios_df.to_hdf(os.path.join(save_dir, filename),"MW", mode="w")

            #     ## save weights
            #     filename = "{}orig_weights_{}.h5".format(n_scenarios, grid_name)
            #     weights_df.to_hdf(os.path.join(save_dir, filename),"MW", mode="w")




        # flatten chunks back to one list
    #     new_file_names = [file_name for chunk in new_file_names_chunks for file_name in chunk]
    #     # now find the max in the list of maxes
    #     max_value_file = max(new_file_names,key=itemgetter(0))

    #     print('max value file: {} '.format(max_value_file))
    #     t1 = time.time()
    #     print('elapsed time = {}'.format(t1-t0))




if __name__ == "__main__":
    main()





