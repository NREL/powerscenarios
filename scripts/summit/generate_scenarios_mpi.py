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

pd.set_option('display.width', None)

# Import the module
comm = MPI.COMM_WORLD
my_mpi_rank = comm.Get_rank()
comm_size = comm.Get_size()

if my_mpi_rank == 0:
    start_time = time.time()

# grid_name = "ACTIVSg200"  # TAMU 200 bus case
grid_name = "ACTIVSg2000"  # TAMU 2000 bus case

data_dir = "../../data/grid-data/"
aux_file_name = os.path.join(data_dir, grid_name, grid_name + ".aux")

# parse original .aux file and return dataframes for buses, generators, and wind generators
# here, we need .aux files because those are the only ones with Latitute/Longitude information
parser = Parser()
bus_df, gen_df, wind_gen_df = parser.parse_tamu_aux(aux_file_name)

grid = Grid(grid_name, bus_df, gen_df, wind_gen_df)
grid.retrieve_wind_sites(method="simple proximity")

grid.make_tables(actuals_start=pd.Timestamp("2007-01-01 00:00:00", tz="utc"),
                 actuals_end=pd.Timestamp("2007-12-31 23:55:00", tz="utc"),
                 scenarios_start=pd.Timestamp("2008-01-01 00:00:00", tz="utc"),
                 scenarios_end=pd.Timestamp("2013-12-31 23:55:00", tz="utc"),
                )
## save after retrieve_wind_sites and make_tables



# for actuals, make year you want
grid.actuals.index = grid.actuals.index.map(lambda t: t.replace(year=2020))

# a few timestamps timestamp
sim_timestamps = [pd.Timestamp("2020-09-23 07:00:00+0000", tz="UTC"),]
                  # pd.Timestamp("2020-09-23 07:05:00+0000", tz="UTC")]

# other parameters
# sampling_method="monte carlo"
sampling_method = "importance"
# fidelity = "exago_file"
fidelity = "exago_lib"
n_scenarios = 1
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
    random_seed = np.random.randint(2 ** 31 - 1)
    # random_seed = 594081473
    # print("random_seed = {}".format(random_seed))
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
        pricing_scen_ct = 10000,
        mpi_comm = comm
    )
    # all_scenarios_df=pd.concat([all_scenarios_df,scenarios_df])
    all_scenarios_df.loc[sim_timestamp] = scenarios_df

    # all_weights_df=pd.concat([all_weights_df,weights_df])
    all_weights_df.loc[sim_timestamp] = weights_df.loc[sim_timestamp]


# copy wanted actuals
all_actuals_df = grid.actuals.loc[sim_timestamps].drop("TotalPower", axis=1).copy()
# match index name to all_scenarios_df
all_actuals_df.index.name = "sim_timestamp"

for i in range(comm_size):
    if my_mpi_rank == 0:
        # print("\nall_actuals_df:")
        # all_actuals_df
        print("rank = ", my_mpi_rank)
        # print("\nall_scenarios_df:")
        print(all_scenarios_df)
        # print("\nall_weights_df:")
        # all_weights_df
    comm.Barrier()


# print("\nall_scenarios_df:")
# print(all_scenarios_df)

comm.Barrier()
if my_mpi_rank == 0:
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("\ntime_elapsed = ", time_elapsed)
