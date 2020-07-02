from __future__ import print_function
import logging
import pandas as pd
import numpy as np
import sys

# should this be imported only as needed (in retrieve_wind_sites, retrieve_wtk_data,)
import os

logging.basicConfig()


class Parser(object):
    """ Class for parsing various grid files 
            TAMU files from: https://electricgrids.engr.tamu.edu/electric-grid-test-cases/
            .aux, .m,
            RTS-GMLC files from:  
    """

    blet = "ble"

    def __init__(self):
        pass

    def remove_spaces_between_quotes(self, line):
        """Internal function to remove spaces in a line if they appear between "" while parsing .aux files for TAMU grids
        
        Required Args:
            line - (String) a line to be modified
        Returns:
            modified line (String)
            
            
        E.g. line:
        '1 "CREVE COEUR 0" 115.00000000000000000000 "NO "'
        turns into:
        '1 "CREVECOEUR0" 115.00000000000000000000 "NO"'
        
        """

        quotes_start = False
        quotes_end = False

        new_line = ""

        for char in line:
            if char == '"' and not quotes_start:
                quotes_start = True
                quotes_end = False
                new_line += char
                continue

            if quotes_start and char == '"':
                quotes_end = True
                quotes_start = False
                new_line += char
                continue

            if quotes_start and not quotes_end:
                if char == " ":
                    new_line += ""
                else:
                    new_line += char

            else:
                new_line += char

        return new_line

    def read_bus_table(self, grid_file_name):
        """ function to read Bus table from TAMU grid .aux file
        
        Required Args:
            grid_file_name - (String) path to .aux file
        Returns:
            pandas DataFrame with columns: BusNum, Latitude, Longitude, Zone, 
            
        Modify source to add other columns.
        """

        table_name = "Bus"
        # when reading .aux end line string is different for different versions of Python
        if sys.version_info[0] == 2:
            end_line_str = "\r\n"
        elif sys.version_info[0] == 3:
            end_line_str = "\n"

        with open(grid_file_name) as f:
            lines = f.readlines()
            # reads all lines in a file at once, for large files can cause memory problems
            # f.readline() reads just one line at a time

        # df = pd.DataFrame()
        table_start = False
        rows_start = False
        rows = []

        for n, line in enumerate(lines):
            words = line.split(" ")
            if words[0] == "DATA" and words[1][1:-1] == table_name:
                # new table table_name starts
                table_start = True
                continue
            if table_start and line == "{" + end_line_str:

                rows_start = True
                continue
            if rows_start and line != "}" + end_line_str:
                line = self.remove_spaces_between_quotes(line)
                line = line.replace(end_line_str, "")
                row = line.split()
                # this is where you pick what columns you want (and set their types)
                try:
                    rows.append(
                        {
                            "BusNum": int(row[0]),
                            "BusName": row[1][1:-1],
                            "Latitude": float(row[14]),
                            "Longitude": float(row[15]),
                            "Zone": int(row[10]),
                        }
                    )
                except Exception as e:
                    print("data conversion error: " + str(e))
                    break

            elif rows_start and line == "}" + end_line_str:
                break

        # create df out of a list of dicts
        df = pd.DataFrame(rows)

        if df.empty:
            print("dataframe empty, try another end_line_str")

        return df

    def read_gen_table(self, grid_file_name):
        """ function to read Gen table from TAMU grid .aux file
        
        Required Args:
            grid_file_name - (String) path to .aux file
        Returns:
            pandas DataFrame with columns: BusNum, GenMWMax, GenMWMin, GenWindPowerFactor, GenFuelType
            
        Modify source to add other columns.
        """

        # file_name = "../grids/ACTIVSg200/ACTIVSg200.aux"
        # file_name = "../grids/ACTIVSg2000/ACTIVSg2000.aux"
        table_name = "Gen"

        # when reading .aux end line string is different for different versions of Python

        if sys.version_info[0] == 2:
            end_line_str = "\r\n"
        elif sys.version_info[0] == 3:
            end_line_str = "\n"

        with open(grid_file_name) as f:
            lines = f.readlines()

        # this is tricky, at first lines ended with \r\n, now it is just \n (maybe because I saved it with text editor??)
        # end_line_str = '\r\n'
        end_line_str = "\n"

        # df = pd.DataFrame()
        table_start = False
        rows_start = False
        rows = []

        for n, line in enumerate(lines):
            words = line.split(" ")
            if words[0] == "DATA" and words[1][1:-1] == table_name:
                # new table table_name starts
                table_start = True
                # column_names = words[2]
                continue
            if table_start and line == "{" + end_line_str:
                rows_start = True
                continue
            if rows_start and line != "}" + end_line_str:
                line = self.remove_spaces_between_quotes(line)
                line = line.replace(end_line_str, "")
                row = line.split()
                # this is where you pick what columns you want (and set their types)
                try:
                    rows.append(
                        {
                            "BusNum": int(row[0]),
                            "GenID": int(row[1][1:-1]),
                            "GenMWMax": float(row[9]),
                            "GenMWMin": float(row[10]),
                            "GenWindPowerFactor": float(row[18]),
                            "GenFuelType": row[56][1:-1],
                        }
                    )
                except Exception as e:
                    print("data conversion error: " + str(e))
                    print("line nr = {}".format(n))
                    print(row[0])
                    break

            elif rows_start and line == "}" + end_line_str:
                break

        df = pd.DataFrame(rows)

        if df.empty:
            print("dataframe empty, try another end_line_str")

        return df

    def parse_tamu_aux(self, aux_file_name):
        """ function to parse TAMU grid .aux files
        Required Args:
            aux_file_name - (String) path to .aux file
        Returns:
            bus_df (pandas DataFrame) with columns: BusNum, BusName, Latitude, Longitute
            gen_df (pandas DataFrame) with columns: GenUID, BusNum, GenFuelType, GenMWMax, GenMWMin, GenID, BusName, Latitude, Longitude 
            wind_gen_df (pandas DataFrame) with columns: GenUID(index), BusNum, GenFuelType, GenMWMax, GenMWMin, GenID, BusName, Latitude, Longitude 

        Modify source if other columns are needed
        """
        # print('parsing {}\n'.format(aux_file_name))

        # read Bus table
        bus_df = self.read_bus_table(aux_file_name)

        # read Gen table
        gen_df = self.read_gen_table(aux_file_name)

        # create unique generator ID (GenUID) column
        gen_df["GenUID"] = (
            gen_df["BusNum"].apply(str)
            + "_"
            + gen_df["GenFuelType"]
            + "_"
            + gen_df["GenID"].apply(str)
        )
        # gen_df.head()

        # add Latitude, Longitude, BusName to gen_df from bus_df (i.e. merge gen_df with bus_df on BusNum)
        new_gen_df = gen_df.merge(bus_df, on="BusNum")
        # print('new_gen_df:')
        # new_gen_df.head()
        # new_gen_df.info()

        # create wind generator df (i.e. take only wind generators)
        wind_gen_df = new_gen_df[new_gen_df["GenFuelType"] == "Wind"].copy()
        # reset index to standard
        wind_gen_df.reset_index(inplace=True, drop=True)
        # print('wind_gen_df:')
        # wind_gen_df.head()

        # # take only wind generators
        # wind_gen_df_temp = gen_df[gen_df['GenFuelType']=='Wind'].copy()

        # # merge wind_gen_df  with bus_df on BusNum (inner join by default)
        # # so that we know what bus does wind generator belongs to (and hence, its coordinates)
        # wind_gen_df = pd.merge(wind_gen_df_temp, bus_df, left_on='BusNum', right_on='BusNum')

        # # drop GenFuelType column, since these are all wind generstors now
        # wind_gen_df.drop('GenFuelType', axis=1, inplace=True)

        # print('Done.')

        return bus_df, new_gen_df, wind_gen_df

    def parse_rts_csvs(self, bus_csv_file_name, gen_csv_file_name):
        """ function to parse RTS grid .csv files (bus.csv and gen.csv)
        Required Args:
            bus_csv_file_name - (String) path to RTS bus.csv file
            gen_csv_file_name - (String) path to RTS gen.csv file
        Returns:
            bus_df (pandas DataFrame) 
            gen_df (pandas DataFrame) 
            wind_gen_df (pandas DataFrame) 
        
        """

        # print('parsing files: {}, {}\n'.format(bus_csv_file_name, gen_csv_file_name))

        # read original bus.csv, take what you need, drop the rest, and rename columns to match TAMU convention
        bus_df = pd.read_csv(bus_csv_file_name)

        ## all we really need is Bus ID, lat, lng
        #bus_df = bus_df[["Bus ID", "lat", "lng"]].copy()

        # rename columns to match TAMU convention
        bus_df.rename(
            columns={"Bus ID": "BusNum", "lat": "Latitude", "lng": "Longitude"},
            inplace=True,
        )

        # read original gen.csv, take what you need, and save as .csv
        gen_df = pd.read_csv(gen_csv_file_name)

        ## all we really need is gen ID, Fuel, PMax MW
        ## gen_df = gen_df[['Bus ID','Fuel','PMax MW', 'GEN UID']].copy()
        #gen_df = gen_df[["Bus ID", "Fuel", "PMax MW"]].copy()

        # rename columns to match my TAMU convention
        gen_df.rename(
            columns={
                "Bus ID": "BusNum",
                "Fuel": "GenFuelType",
                "PMax MW": "GenMWMax",
                "GEN UID": "GenUID",
            },
            inplace=True,
        )
        # gen_df.rename(columns={'Bus ID':'BusNum', 'Fuel':'GenFuelType', 'PMax MW':'GenMWMax' , }, inplace=True)

        # take wind generators and solar generators (turn them into wind)
        # wind generators
        wind_gen_df = gen_df.loc[gen_df["GenFuelType"] == "Wind"].copy()

        # add lat/long info to wind_gen_df 
        wind_gen_df.set_index("BusNum", inplace=True)
        bus_df.set_index("BusNum", inplace = True)
        wind_gen_df = pd.concat([wind_gen_df,bus_df.loc[wind_gen_df.index][["Latitude","Longitude"]]], axis=1)
        wind_gen_df.reset_index(inplace=True)
        bus_df.reset_index(inplace=True)

        return bus_df, gen_df, wind_gen_df




    ################ parsing of TAMU .m files

    # thats how .m file tables look

    # %% Change Table
    # %       label   prob    table   row     col     chgtype newval
    # chgtab = [
    #         1       0       CT_TAREALOAD    2       CT_LOAD_ALL_P   CT_REP  412.8;
    #         1       0       CT_TAREALOAD    3       CT_LOAD_ALL_P   CT_REP  144.2;
    #         1       0       CT_TAREALOAD    4       CT_LOAD_ALL_P   CT_REP  204.5;

    def read_m_table(
        self, filename, table_name=None, column_names=None, numeric_columns=None
    ):
        """ function to read tables in .m files
        Required Args:
            filename - (str) path to .m file
            table_name - (str) name of the table in .m file before equals sign (e.g. 'chgtab', 'mpc.bus', etc)
            column_names - (list of strings) column names as in .m file (e.g. ['label', 'prob', 'table', 'row', 'col', 'chgtype', 'newval'])
            numeric_columns (list of strings) a subset of column_names that have numeric types (e.g. numeric_columns = ['label', 'prob', 'row', 'newval'])
        
        Returns:
            data_df - (pandas.DataFrame) 
        
        """
        with open(filename) as f:
            lines = f.readlines()
            # reads all lines in a file at once, for large files can cause memory problems
            # f.readline() reads just one line at a time

        # df = pd.DataFrame()
        table_start = False
        rows_start = False
        rows = []

        for n, line in enumerate(lines):
            if line == table_name + " = [\n":
                # print("table start")
                table_start = True
                continue
            if table_start and line != "];\n":
                words = line[1:].replace(";\n", "").split("\t")
                data = dict(zip(column_names, words))
                rows.append(data)
            elif table_start and line == "];\n":
                break

        if not table_start:
            raise ValueError("No table: " + table_name + " in file: " + filename)

        data_df = pd.DataFrame(rows, columns=column_names)

        # change selected columns to numeric values (int or float)
        for column_name in numeric_columns:
            data_df[column_name] = pd.to_numeric(data_df[column_name])

        return data_df

    def read_m_series(
        self, filename, series_name=None,
    ):
        """ function to read series (values of just one column) in .m files; use to augment bus and gen tables
            mpc.gentype, mpc.genfuel, and mpc.bus_name are given separately (because those are not numeric columns?) 
        Required Args:
            filename - (str) path to .m file
            table_name - (str) name of the table in .m file before equals sign (e.g. 'mpc.gentype', 'mpc.bus_name', or mpc.genfuel)
        Returns:
            data_s - (pandas.Series) 

        """
        with open(filename) as f:
            lines = f.readlines()
            # reads all lines in a file at once, for large files can cause memory problems
            # f.readline() reads just one line at a time

        # df = pd.DataFrame()
        series_start = False
        rows = []

        for n, line in enumerate(lines):
            if line == series_name + " = {\n":
                # print("table start")
                series_start = True
                continue
            if series_start and line != "};\n":
                # print(n)
                # print(line)
                word = word = line.replace("\t'", "").replace("';\n", "")
                # print(word)
                rows.append(word)
            # if n==718: break
            elif series_start and line == "};\n":
                break

        if not series_start:
            raise ValueError("No table: " + table_name + " in file: " + filename)

        data_s = pd.Series(rows,)

        #     # change selected columns to numeric values (int or float)
        #     for column_name in numeric_columns:
        #         data_df[column_name] = pd.to_numeric(data_df[column_name])

        return data_s

    def parse_tamu_m(self, case_m_file=None, scenarios_m_file=None):
        """parse_tamu_m parses TAMU grid's .m files and returns all atables as dataframes
        Required Args:
            case_m_file - (str) path to case*.m file for bus, gen, and branch tables (e.g. '/grid-data/ACTIVSg200/case_ACTIVSg200.m')
            scenarios_m_file - (str) path to scenarios*.m file for chgtab table needed for load timeseries (e.g. '/grid-data/ACTIVSg200/scenarios_ACTIVSg200.m')
        Returns:
            bus_df, gen_df, branch_df, chgtab_df - (pd.DataFrame) buses, generators, branches, and change table (load profiles by zone)
        """

        # read bus table from case_m_file
        table_name = "mpc.bus"
        column_names = [
            "bus_i",
            "type",
            "Pd",
            "Qd",
            "Gs",
            "Bs",
            "area",
            "Vm",
            "Va",
            "baseKV",
            "zone",
            "Vmax",
            "Vmin",
            "lam_P",
            "lam_Q",
            "mu_Vmax",
            "mu_Vmin",
        ]
        # this table is entirely numeric
        numeric_columns = column_names
        bus_df = self.read_m_table(
            case_m_file,
            table_name=table_name,
            column_names=column_names,
            numeric_columns=numeric_columns,
        )

        # read gen table from case_m_file
        table_name = "mpc.gen"
        column_names = [
            "bus",
            "Pg",
            "Qg",
            "Qmax",
            "Qmin",
            "Vg",
            "mBase",
            "status",
            "Pmax",
            "Pmin",
            "Pc1",
            "Pc2",
            "Qc1min",
            "Qc1max",
            "Qc2min",
            "Qc2max",
            "ramp_agc",
            "ramp_10",
            "ramp_30",
            "ramp_q",
            "apf",
            "mu_Pmax",
            "mu_Pmin",
            "mu_Qmax",
            "mu_Qmin",
        ]
        # this table is entirely numeric
        numeric_columns = column_names

        gen_df = self.read_m_table(
            case_m_file,
            table_name=table_name,
            column_names=column_names,
            numeric_columns=numeric_columns,
        )

        # read branch table from scenarios_m_file
        table_name = "mpc.branch"
        column_names = [
            "fbus",
            "tbus",
            "r",
            "x",
            "b",
            "rateA",
            "rateB",
            "rateC",
            "ratio",
            "angle",
            "status",
            "angmin",
            "angmax",
            "Pf",
            "Qf",
            "Pt",
            "Qt",
            "mu_Sf",
            "mu_St",
            "mu_angmin",
            "mu_angmax",
        ]

        # this table is entirely numeric
        numeric_columns = column_names

        branch_df = self.read_m_table(
            case_m_file,
            table_name=table_name,
            column_names=column_names,
            numeric_columns=numeric_columns,
        )

        # read chgtab table from scenarios_m_file
        table_name = "chgtab"
        column_names = ["label", "prob", "table", "row", "col", "chgtype", "newval"]
        numeric_columns = ["label", "prob", "row", "newval"]

        chgtab_df = self.read_m_table(
            scenarios_m_file,
            table_name=table_name,
            column_names=column_names,
            numeric_columns=numeric_columns,
        )
        return bus_df, gen_df, branch_df, chgtab_df

    #####################################################################a

    #####################################################################
    def parse_tamu_load_csv(
        self, filename, **kwargs,
    ):
        """
        Parse TAMU load .csv files: TAMU 2K and above have power timeseries (real and reactive power)
        For TAMU 200 and 500 still have to use .m files to get load
        Required Args:
            filename - (str) path to .csv file
            e.g. filename = "../grids-data/ACTIVSg2000/ACTIVISg2000_load_time_series_MVAR.csv"
        Returns:
            df - (pandas.DataFrame) a year of load timeseries, linearly interpolated every 5 min and aggregated by bus
        """
        # read .csv, parse date and time columns, combine to one
        df = pd.read_csv(filename, skiprows=1, parse_dates={"DateTime": [0, 1]})

        # set index as DateTime column
        df.set_index("DateTime", inplace=True)

        # not sure about this????
        # # localize to UTC
        # df = df.tz_localize("UTC", axis=0)

        # drop redundant columns: ["Total MW Load", "Total Mvar Load", "Num Load"]
        # Total MW Load - we can always sum all columns
        # Total Mvar Load - all zeroes, Mvar load is in different file
        # df["Num Load"].unique() -> 1350 # "Num Load" has just one value of 1350 that repeats for all rows - redundant!
        df.drop(["Total MW Load", "Total Mvar Load", "Num Load"], axis=1, inplace=True)
        # rename columns: "Bus 1001 #1 MW" -> 1001
        old_names = df.columns.to_list()
        # old_names
        new_names = [x.split(" ")[1] for x in old_names]
        # new_names
        rename_dict = dict(zip(old_names, new_names))
        df.rename(columns=rename_dict, inplace=True)
        df.head()

        # add columns with the same bus #, i.e. aggregate load by bus
        df = df.groupby(lambda x: x, axis=1).sum()

        # interpolate linearly every five min
        df_resampled = df.resample("5min").interpolate("linear")

        ## this should be done outside
        #         # take just select timestamps
        #         start_of_data = pd.Timestamp("2016-07-01 00:00:00")
        #         end_of_data = pd.Timestamp("2016-07-07 23:55:00")
        #        df_part = df_resampled.loc[start_of_data:end_of_data].copy()

        return df_resampled

    # read any table from .aux file
    def read_aux_table(self, grid_file_name, table_name="Bus", branch_nr=1, **kwargs):
        """ function to read table from TAMU grid .aux file
        
        Required Args:
            grid_file_name - (str) path to .aux file
            table_name - (str) one of the following: 'Bus', 'Gen', 'Branch'
            brsnch_nr - (int)  branch table number: 1 or 2 (there are 2 Branch tables: lines and transformers)
        Returns:
            table_df - (pd.DataFrame) table with all available columns in .aux file
            
        """
        if table_name == "Bus":
            all_cols = [
                "BusNum",
                "BusName",
                "BusNomVolt",
                "BusSlack",
                "BusB:1",
                "BusG:1",
                "BusPUVolt",
                "BusAngle",
                "DCLossMultiplier",
                "AreaNum",
                "ZoneNum",
                "BANumber",
                "OwnerNum",
                "SubNum",
                "Latitude:1",
                "Longitude:1",
                "BusMonEle",
                "LSName",
                "BusVoltLim",
                "BusVoltLimLow:1",
                "BusVoltLimLow:2",
                "BusVoltLimLow:3",
                "BusVoltLimLow:4",
                "BusVoltLimHigh:1",
                "BusVoltLimHigh:2",
                "BusVoltLimHigh:3",
                "BusVoltLimHigh:4",
                "Latitude",
                "Longitude",
                "TopologyBusType",
                "Priority",
                "EMSType",
                "EMSDeviceID",
                "DataMaintainerAssign",
                "AllLabels",
                "GICConductance",
            ]
        elif table_name == "Gen":
            all_cols = [
                "BusNum",
                "GenID",
                "GenStatus",
                "GenVoltSet",
                "GenRegNum",
                "GenRMPCT",
                "GenAGCAble",
                "GenParFac",
                "GenMWSetPoint",
                "GenMWMax",
                "GenMWMin",
                "GenEnforceMWLimits",
                "GenAVRAble",
                "GenMvrSetPoint",
                "GenMVRMax",
                "GenMVRMin",
                "GenUseCapCurve",
                "GenWindControlMode",
                "GenWindPowerFactor",
                "GenUseLDCRCC",
                "GenRLDCRCC",
                "GenXLDCRCC",
                "GenMVABase",
                "GenZR",
                "GenZX",
                "GenStepR",
                "GenStepX",
                "GenStepTap",
                "TSGovRespLimit",
                "GenUnitType:1",
                "AreaNum",
                "ZoneNum",
                "BANumber",
                "OwnerNum",
                "OwnPercent",
                "OwnerNum:1",
                "OwnPercent:1",
                "OwnerNum:2",
                "OwnPercent:2",
                "OwnerNum:3",
                "OwnPercent:3",
                "OwnerNum:4",
                "OwnPercent:4",
                "OwnerNum:5",
                "OwnPercent:5",
                "OwnerNum:6",
                "OwnPercent:6",
                "OwnerNum:7",
                "OwnPercent:7",
                "EMSType",
                "EMSDeviceID",
                "DataMaintainerAssign",
                "AllLabels",
                "GenUnitType",
                "GenTotalFixedCosts",
                "GenCostModel",
                "GenFuelType",
                "GenFuelCost",
                "GenFixedCost",
                "GenIOD",
                "GenIOC",
                "GenIOB",
            ]
        elif table_name == "Branch" and branch_nr == 1:
            all_cols = [
                "BusNum",
                "BusNum:1",
                "LineCircuit",
                "BranchDeviceType",
                "ConsolidateBranch",
                "LineStatus",
                "NormLineStatus",
                "SeriesCapStatus",
                "LineMeter:1",
                "LineR",
                "LineX",
                "LineC",
                "LineG",
                "LineLength",
                "LineMonEle",
                "LSName",
                "LineAMVA",
                "LineAMVA:1",
                "LineAMVA:2",
                "LineAMVA:3",
                "LineAMVA:4",
                "LineAMVA:5",
                "LineAMVA:6",
                "LineAMVA:7",
                "LineAMVA:8",
                "LineAMVA:9",
                "LineAMVA:10",
                "LineAMVA:11",
                "LineAMVA:12",
                "LineAMVA:13",
                "LineAMVA:14",
                "OwnerNum",
                "OwnPercent",
                "OwnerNum:1",
                "OwnPercent:1",
                "OwnerNum:2",
                "OwnPercent:2",
                "OwnerNum:3",
                "OwnPercent:3",
                "OwnerNum:4",
                "OwnPercent:4",
                "OwnerNum:5",
                "OwnPercent:5",
                "OwnerNum:6",
                "OwnPercent:6",
                "OwnerNum:7",
                "OwnPercent:7",
                "EMSType",
                "EMSDeviceID",
                "EMSDeviceID:1",
                "EMSType:1",
                "EMSDeviceID:2",
                "EMSDeviceID:3",
                "DataMaintainerAssign",
                "AllLabels",
            ]
        elif table_name == "Branch" and branch_nr == 2:
            all_cols = [
                "BusNum",
                "BusNum:1",
                "LineCircuit",
                "BranchDeviceType",
                "LineStatus",
                "NormLineStatus",
                "SeriesCapStatus",
                "LineMeter:1",
                "LineXFType",
                "XFAuto",
                "XFRegBus",
                "XFUseLDCRCC",
                "XFRLDCRCC",
                "XFXLDCRCC",
                "XFRegMax",
                "XFRegMin",
                "XFRegTargetType",
                "XFMVABase",
                "XFNominalKV",
                "XFNominalKV:1",
                "LineR:1",
                "LineX:1",
                "LineG:1",
                "LineC:1",
                "XfrmerMagnetizingG:1",
                "XfrmerMagnetizingB:1",
                "XFFixedTap",
                "XFFixedTap:1",
                "XFTapMax:1",
                "XFTapMin:1",
                "XFStep:1",
                "LineTap:1",
                "LinePhase",
                "XFTableNum",
                "LineLength",
                "LineMonEle",
                "LSName",
                "LineAMVA",
                "LineAMVA:1",
                "LineAMVA:2",
                "LineAMVA:3",
                "LineAMVA:4",
                "LineAMVA:5",
                "LineAMVA:6",
                "LineAMVA:7",
                "LineAMVA:8",
                "LineAMVA:9",
                "LineAMVA:10",
                "LineAMVA:11",
                "LineAMVA:12",
                "LineAMVA:13",
                "LineAMVA:14",
                "OwnerNum",
                "OwnPercent",
                "OwnerNum:1",
                "OwnPercent:1",
                "OwnerNum:2",
                "OwnPercent:2",
                "OwnerNum:3",
                "OwnPercent:3",
                "OwnerNum:4",
                "OwnPercent:4",
                "OwnerNum:5",
                "OwnPercent:5",
                "OwnerNum:6",
                "OwnPercent:6",
                "OwnerNum:7",
                "OwnPercent:7",
                "EMSType",
                "EMSDeviceID",
                "EMSDeviceID:1",
                "EMSType:1",
                "EMSDeviceID:2",
                "EMSDeviceID:3",
                "DataMaintainerAssign",
                "AllLabels",
            ]

        # when reading .aux end line string is different for different versions of Python

        if sys.version_info[0] == 2:
            end_line_str = "\r\n"
        elif sys.version_info[0] == 3:
            end_line_str = "\n"

        with open(grid_file_name) as f:
            lines = f.readlines()
            # reads all lines in a file at once, for large files can cause memory problems
            # f.readline() reads just one line at a time

        # df = pd.DataFrame()
        table_start = False
        rows_start = False
        # second_branch_table = False
        rows = []

        for n, line in enumerate(lines):
            words = line.split(" ")

            #         print(n)
            #         print(line)
            #         if n==200:
            #             return row
            if words[0] == "DATA" and words[1][1:-1] == table_name:
                # new table table_name starts
                if table_name != "Branch" or (
                    table_name == "Branch" and branch_nr == 1
                ):
                    table_start = True
                    continue
                elif table_name == "Branch" and branch_nr == 2:
                    branch_nr = 1
                    continue

            if table_start and line == "{" + end_line_str:

                rows_start = True
                continue
            if rows_start and line != "}" + end_line_str:
                line = self.remove_spaces_between_quotes(line)
                line = line.replace(end_line_str, "")
                row = line.split()
                # this is where we take all columns
                try:
                    rows.append(dict(zip(all_cols, row)))
                except Exception as e:
                    print("data conversion error: " + str(e))
                    break

            # end reading of the table: for 'Branch' table ending is different
            elif rows_start and line == "}" + end_line_str:
                break

        # create df out of a list of dicts
        df = pd.DataFrame(rows)

        if df.empty:
            print("dataframe empty, try another end_line_str")

        return df
