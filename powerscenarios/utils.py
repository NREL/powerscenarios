# utils.py
import numpy as np
import pandas as pd
import os

def broadcast_dataframe(df, mpi_comm, root=0):
    """
    This function is used to broadcast a dataframe to all ranks from root. By
    default root is 0 (master)
    """

    rank = mpi_comm.Get_rank()
    if rank == 0:
        # print(df)
        og_idx_name = df.index.name
        df_copy = df.reset_index()
        df_colnames = list(df_copy.columns)
        df_list = df_copy.values.tolist()
    else:
        df_list = None
        og_idx_name = None
        df_colnames = None

    mpi_comm.Barrier()
    og_idx_name = mpi_comm.bcast(og_idx_name, root=root)
    df_colnames = mpi_comm.bcast(df_colnames, root=root)
    df_list = mpi_comm.bcast(df_list, root=0)

    # Finally reconstruct the dataframes on other ranks
    if rank != root:
        df = pd.DataFrame(df_list, columns=df_colnames)
        if og_idx_name is not None: # Only reset index if it has a name
            df.set_index(og_idx_name, inplace=True)


    return df

def save_output(grid, sim_timestamp, actuals_df, scenarios_df, weights_df,
                n_scenarios, save_dir, df_format_type="original",
                file_format="csv"):

    grid_name = grid.name

    if df_format_type == "Shri":
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

    elif df_format_type == "original":
        if file_format == "csv":
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
