import pandas as pd
import numpy as np
import time
import os
# from mpi4py import MPI

# from exago.opflow import OPFLOW
# from exago import config

from powerscenarios.parser import Parser
from powerscenarios.grid_copy import Grid

grid_name = "ACTIVSg200"  # TAMU 200 bus case
# grid_name = "ACTIVSg2000"  # TAMU 2000 bus case

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

## save output to hdf
output_dir = "./output/"

# Wind Sites
filename_wind_sites = "{}_wind_sites.h5".format(grid_name)
grid.wind_sites.to_hdf(os.path.join(output_dir,filename_wind_sites),'MW',  mode='w')
print("saving grid.wind_sites to {}".format(os.path.join(output_dir,filename_wind_sites)))

# Scenarios
filename_scen_table = "{}_scenarios_table.h5".format(grid_name)
grid.scenarios.to_hdf(os.path.join(output_dir,filename_scen_table),'MW',  mode='w')
print("saving grid.scenarios to {}".format(os.path.join(output_dir,filename_scen_table)))


# Actuals
filename_act_table = "{}_actuals_table.h5".format(grid_name)
grid.actuals.to_hdf(os.path.join(output_dir,filename_act_table),'MW',  mode='w')
print("saving grid.actuals to {}".format(os.path.join(output_dir,filename_act_table)))

# filename = "wind_sites_df.csv"
# grid.wind_sites.to_csv(filename)
# filename = "actuals_df.csv"
# grid.actuals.to_csv(filename)
# filename = "scenarios_df.csv"
# grid.scenarios.to_csv(filename)
