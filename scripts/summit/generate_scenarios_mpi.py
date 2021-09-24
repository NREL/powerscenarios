import pandas as pd
import numpy as np
import time
import os
from mpi4py import MPI
# from exago.opflow import OPFLOW
# from exago import config

from powerscenarios.parser import Parser
from powerscenarios.grid_copy import Grid
from powerscenarios.costs.exago.exago_lib import ExaGO_Lib
import powerscenarios.utils as utils

pd.set_option('display.width', None)

# Import the module
comm = MPI.COMM_WORLD
my_mpi_rank = comm.Get_rank()
comm_size = comm.Get_size()

read_grid_data = True

if my_mpi_rank == 0:
    start_time = time.time()

# grid_name = "ACTIVSg200"  # TAMU 200 bus case
grid_name = "ACTIVSg2000"  # TAMU 2000 bus case

if grid_name == "ACTIVSg200":
    sim_timestamps = [pd.Timestamp("2020-12-20 08:00:00+0000", tz="UTC"),]
elif grid_name == "ACTIVSg2000":
    sim_timestamps = [pd.Timestamp("2020-03-22 10:45:00+0000", tz="UTC"),]
else:
    raise NotImplementedError

data_dir = "../../data/grid-data/"
aux_file_name = os.path.join(data_dir, grid_name, grid_name + ".aux")

if my_mpi_rank == 0:
    random_seed = np.random.randint(2 ** 31 - 1)
    # parse original .aux file and return dataframes for buses, generators, and wind generators
    # here, we need .aux files because those are the only ones with Latitute/Longitude information
    parser = Parser()
    bus_df, gen_df, wind_gen_df = parser.parse_tamu_aux(aux_file_name)
else:
    random_seed = None
    bus_df = None
    gen_df = None
    wind_gen_df = None

bus_df = utils.broadcast_dataframe(bus_df, comm)
gen_df = utils.broadcast_dataframe(gen_df, comm)
wind_gen_df = utils.broadcast_dataframe(wind_gen_df, comm)
random_seed = comm.bcast(random_seed, root=0)
comm.Barrier()

if my_mpi_rank == 0:
    grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)

    if read_grid_data:
        print("Reading h5s for {0}".format(grid_name))
        # Wind sites
        filename_wind_sites = "./output/{0}_wind_sites.h5".format(grid_name)
        grid.wind_sites = pd.read_hdf(filename_wind_sites)
        # Scenario Table
        filename_scen_table = "./output/{}_scenarios_table.h5".format(grid_name)
        grid.scenarios = pd.read_hdf(filename_scen_table)
        # Actuals Table
        filename_act_table = "./output/{}_actuals_table.h5".format(grid_name)
        grid.actuals = pd.read_hdf(filename_act_table)

    else:
        grid.retrieve_wind_sites(method="simple proximity")
        grid.make_tables(actuals_start=pd.Timestamp("2007-01-01 00:00:00", tz="utc"),
                         actuals_end=pd.Timestamp("2007-12-31 23:55:00", tz="utc"),
                         scenarios_start=pd.Timestamp("2008-01-01 00:00:00", tz="utc"),
                         scenarios_end=pd.Timestamp("2013-12-31 23:55:00", tz="utc"),
                        )
    # for actuals, make year you want
    grid.actuals.index = grid.actuals.index.map(lambda t: t.replace(year=2020))

    actuals_df = grid.actuals
    scenarios_df = grid.scenarios
else:
    grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)
    actuals_df = None
    scenarios_df = None

comm.Barrier()
actuals_df = utils.broadcast_dataframe(actuals_df, comm)
scenarios_df = utils.broadcast_dataframe(scenarios_df, comm)
if my_mpi_rank != 0:
    grid.actuals = actuals_df
    grid.scenarios = scenarios_df

# # a few timestamps timestamp
# sim_timestamps = [pd.Timestamp("2020-03-22 10:45:00+0000", tz="UTC"),]
#                   # pd.Timestamp("2020-09-23 07:05:00+0000", tz="UTC")]

# other parameters
sampling_method = "importance"
fidelity = "exago_lib"
n_scenarios = 10
n_periods = 1

########################################################

all_weights_df = pd.DataFrame(index=sim_timestamps, columns=range(1, n_scenarios + 1))

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

index = pd.MultiIndex.from_arrays(
    [a1, a2, a3], names=["sim_timestamp", "scenario_nr", "period_timestamp"]
)
all_scenarios_df = pd.DataFrame(
    index=index, columns=grid.wind_generators["GenUID"].values
)


for sim_timestamp in sim_timestamps:
    # print("sim_timestamp = {}".format(sim_timestamp))

    # random_seed = 594081473
    scenarios_df, weights_df, p_bin, cost_n = grid.generate_wind_scenarios(
        sim_timestamp,
        power_quantiles=[0.0, 0.1, 0.9, 1.0],
        sampling_method=sampling_method,
        fidelity=fidelity,
        n_scenarios=n_scenarios,
        n_periods=n_periods,
        # random_seed=6,
        random_seed=random_seed,
        output_format=0,
        pricing_scen_ct = 300000,
        mpi_comm = comm
    )
    all_scenarios_df.loc[sim_timestamp] = scenarios_df
    all_weights_df.loc[sim_timestamp] = weights_df.loc[sim_timestamp]


# copy wanted actuals
all_actuals_df = grid.actuals.loc[sim_timestamps].drop("TotalPower", axis=1).copy()
# match index name to all_scenarios_df
all_actuals_df.index.name = "sim_timestamp"

if my_mpi_rank == 0:
    # Save the scenarios in the requisite format
    save_dir = "./output/"
    utils.save_output(grid, sim_timestamp, all_actuals_df, all_scenarios_df,
                      all_weights_df, n_scenarios, save_dir,
                      df_format_type="Shri")

    cost_n_fname = "{0}_cost_n.csv".format(grid_name)
    cost_n.to_csv(os.path.join(save_dir, cost_n_fname))

comm.Barrier()
if my_mpi_rank == 0:
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("\ntime_elapsed = ", time_elapsed)
