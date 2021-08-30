# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time
import re
import subprocess
import warnings
from os import path
import shutil

from powerscenarios.costs.abstract_fidelity import AbstractCostingFidelity
from powerscenarios.costs.exago.matpowerhandler import MatpowerHandler
import os
from pathlib import Path

exago_options_dir = path.dirname(__file__) # Figure out if there is a more elegant implementation

class ExaGO_File(AbstractCostingFidelity):
    """
    This class contains the wrapper for calling OPFLOW from within Powerscenaios
    """
    def __init__(self,
                 n_scenarios, # Number of scenarios we actually want in our final csv file
                 n_periods,
                 loss_of_load_cost,
                 spilled_wind_cost,
                 scenarios_df,
                 p_bin,
                 total_power_t0,
                 WTK_DATA_PRECISION=6,
                 nscen_priced=1):

        AbstractCostingFidelity.__init__(self,
                                         n_scenarios,
                                         n_periods,
                                         loss_of_load_cost,
                                         spilled_wind_cost,
                                         scenarios_df,
                                         p_bin,
                                         total_power_t0,
                                         WTK_DATA_PRECISION=6)

        self.grid_name = "ACTIVSg200"
        self.opflow_options_dict = {'job_launcher' : 'mpirun',
                                   'opflow_solver' : 'IPOPT',
                                   'matpower_file' : "{0}.m".format(self.grid_name),
                                   'output_path' : "opflowout.m",
                                  }
        self.sopflow_options_dict = {'job_launcher' : self.opflow_options_dict['job_launcher'],
                                    'n_mpi_procs' : '8',
                                    'solver' : 'EMPAR',
                                    'nscenarios' : '{0}'.format(nscen_priced), # This needs to be a string as the function runs a bash command
                                   }

        self._create_ego_object()

        return

    def _create_ego_object(self):
        # Data files for creating the file based exago object
        base_dir = Path(__file__).parents[3]
        grid_data_dir = os.path.join(base_dir,"data","grid-data")
        #print("base_dir: {}".format(base_dir))
        #print("grid_data_dir: {}".format(grid_data_dir))

        network_file = os.path.join(grid_data_dir,"{0}/case_{0}.m".format(self.grid_name))
        grid_aux_file =os.path.join(grid_data_dir,"{0}/{0}.aux".format(self.grid_name))
        #print("network_file: {}".format(network_file))
        #print("grid_aux_file: {}".format(grid_aux_file))

        network_file = os.path.join(grid_data_dir,"{0}/case_{0}.m".format(self.grid_name))
        #grid_aux_file =os.path.join(grid_data_dir,"{0}/{0}.aux".format(self.grid_name))

        #network_file = "/Users/kpanda/UserApps/powerscenarios/data/grid-data/{0}/case_{0}.m".format(self.grid_name)

        load_dir = None # "/Users/kpanda/UserApps/powerscenarios/data/load-data"
        real_load_file = None # "/Users/kpanda/UserApps/powerscenarios/data/load-data/{0}_loadP.csv".format(self.grid_name)
        reactive_load_file = None # "/Users/kpanda/UserApps/powerscenarios/data/load-data/{0}_loadQ.csv".format(self.grid_name)

        self.ego = ExaGO(network_file, load_dir, self.grid_name, real_load_file, reactive_load_file)
        self.ego._cleanup() # Lets clean up the file based implementation.

        return


    def compute_scenario_cost(self,
                              actuals_df,
                              binned_scenarios_df,
                              start_time,
                              random_seed=np.random.randint(2 ** 31 - 1)):

        stop_time = start_time # For now

        # Create Persistence Forecast
        persistence_wind_fcst_df = actuals_df.loc[start_time:stop_time,:].copy().drop(columns=["TotalPower"])
        persistence_wind_fcst_df.index = persistence_wind_fcst_df.index + pd.Timedelta(minutes=5.0)
        # display(persistence_wind_fcst_df)
        # There are some dummy/unused variables currently being used in the funciton
        # calls that we will set to None.
        pv_fcst_df = None
        prev_set_points = None
        n_periods = 1
        step = 5.0
        (base_cost, set_points) = self.ego.base_cost(start_time,
                                           pv_fcst_df, # Currently unused
                                           persistence_wind_fcst_df,
                                           prev_set_points, # Currently unused
                                           n_periods, # Currently unused
                                           step, # Currently unused
                                           self.opflow_options_dict,
                                           system="Mac"
                                           )

        print("\nbase_cost = ", base_cost, "\n")
        # For the second part now is to restore some of the original data and the
        # set points from the abse cost runs
        gen_df = self.ego.imat.get_table('gen')

        self.ego._restore_org_gen_table()
        idx = self.ego._non_wind_gens(self.ego.gen_type)
        gen_df.loc[idx,'Pg'] = set_points.loc[idx,'Pg']
        gen_df.loc[idx,'Pmin'] = set_points.loc[idx,'Pg']
        gen_df.loc[idx,'Pmax'] = set_points.loc[idx,'Pg']
        gen_df.loc[idx,'Qg'] = set_points.loc[idx,'Qg']
        gen_df.loc[idx,'Qmin'] = set_points.loc[idx,'Qg']
        gen_df.loc[idx,'Qmax'] = set_points.loc[idx,'Qg']

        # Now that we have the base cost, we need to consider the scenarios at a
        # Given timestamps
        # Turn deviations into scenarios
        wind_scen_df = binned_scenarios_df + persistence_wind_fcst_df.loc[:,binned_scenarios_df.columns].values
        for wgen in wind_scen_df.columns:
            # Enforce Pmax on wind scenarios
            wgen_max = self.ego.wind_max.loc[wgen]
            idx = wind_scen_df.loc[:,wgen] > wgen_max
            wind_scen_df.loc[idx,wgen] = wgen_max
            # Enforce Pmin on wind scenarios
            idx = wind_scen_df.loc[:,wgen] < 0.0
            wind_scen_df.loc[idx,wgen] = 0.0

        print('Available Scenarios = ', wind_scen_df.shape[0], ", Requested Scenarios = ", int(self.sopflow_options_dict['nscenarios']))
        assert wind_scen_df.shape[0] >= int(self.sopflow_options_dict['nscenarios'])
        nscen = int(self.sopflow_options_dict['nscenarios']) # wind_scen_df.shape[0]

        # ExaGO parsing of scenarios.csv seems to depend on 'scenario_nr'
        # being the second column so reorder the columns.
        cols = wind_scen_df.columns.tolist()
        cols.insert(0,'scenario_nr')
        wind_scen_df['scenario_nr'] = range(1, wind_scen_df.shape[0]+1)
        wind_scen_df = wind_scen_df.loc[:,cols]
        if wind_scen_df.shape[0] != int(self.sopflow_options_dict['nscenarios']):
            print("Costing {:d} scenarios\n".format(nscen))
            wind_scen_df = wind_scen_df[wind_scen_df['scenario_nr'].between(1, nscen)]

        t1 = time.time()

        matpower_file = 'scen_{}.m'.format(self.grid_name)
        self.ego.imat.write_matpower_file(matpower_file)

        scenario_file = 'scenarios_{}.csv'.format(self.grid_name)
        wind_scen_df.to_csv(scenario_file)

        t2 = time.time()
        system = "Mac"
        if system != "Summit":
            # exago_cmd = [self.sopflow_options_dict['job_launcher'],
            #              '-n',
            #              self.sopflow_options_dict['n_mpi_procs'], # str(1),
            #              self.ego.sopflow_executable,
            #              '-sopflow_solver',
            #              self.sopflow_options_dict['solver'], # 'EMPAR',
            #              '-netfile',
            #              matpower_file,
            #              '-save_output',
            #              '-opflow_include_loadloss_variables',
            #              str(1),
            #              '-opflow_loadloss_penalty',
            #              str(10000.0 / 12.0),
            #              '-sopflow_Ns',
            #              str(self.sopflow_options_dict['nscenarios']), # str(wind_scen_df.shape[0]),
            #              '-scenfile',
            #              scenario_file,
            #              '-opflow_initialization',
            #              'ACPF', # 'MIDPOINT', # 'FROMFILE', # 'ACPF'
            #              '-opflow_include_powerimbalance_variables',
            #              str(1),
            #              '-opflow_powerimbalance_penalty',
            #              str(10000.0 / 12.0),
            #              '-opflow_genbusvoltage',
            #              'VARIABLE_WITHIN_BOUNDS',
            #              ]
            options_file = os.path.join(self.ego.options_dir, "sopflowoptions")
            exago_cmd = [self.sopflow_options_dict['job_launcher'],
                         '-n', self.sopflow_options_dict['n_mpi_procs'],
                         self.ego.sopflow_executable,
                         '-options_file', options_file,
                         '-sopflow_Ns',
                         str(self.sopflow_options_dict['nscenarios']),
                         ]

        else:
            pass
        print("**** ExaGO Command ****\n", exago_cmd, flush=True)
        print("blet")
        exago_res = subprocess.run(exago_cmd,
                                   capture_output=True,
                                   text=True)

        t3 = time.time()

        # print(exago_res.stdout)
        if exago_res.stderr:
            print("************ BEGIN EXAGO ERROR ************")
            print(exago_res.stderr)
            print("************ END EXAGO ERROR ************")
            with open("exago_empar.err", 'w') as err:
                err.write(exago_res.stderr)

        if exago_res.stdout:
            with open("exago_empar.log", 'w') as log:
                log.write(exago_res.stdout)
        #     print("------------ BEGIN EXAGO STDOUT ------------")
        #     print(exago_res.stdout)
        #     print("------------ END EXAGO STDOUT ------------")

        q_cost = np.zeros(nscen)
        for s in range(0,nscen):
            scen_mp_file = 'sopflowout/scen_{:d}.m'
            scen_cost = self.ego._parse_cost(scen_mp_file.format(s))
            if scen_cost < 0.0:
                print("Scenario {:d} cost failed to parse".format(s))
                q_cost[s-1] = 1e-12
            else:
                q_cost[s-1] = scen_cost - base_cost

        t4 = time.time()

        # Zip the numpy array into the timeseries
        cost_n = pd.Series(index=wind_scen_df.index, data=q_cost)
        print("q_cost = ", repr(q_cost))
        # np.savetxt("cost_sys.txt", q_cost)
        print("q_cost max = ", np.amax(q_cost))
        print("q_cost min = ", np.amin(q_cost))

        return cost_n

###############################################

class ExaGO:

    def __init__(self,
                 network_file,
                 load_dir,
                 grid_name,
                 real_load_file=None,
                 reactive_load_file=None,
                 year=2020):

        start_init = time.time()

        self.grid_name = grid_name
        self.network_file = network_file

        # Get the directory where the ExaGO options are located
        self.options_dir = exago_options_dir
        print("options_dir = ", self.options_dir)

        start = time.time()
        self.imat = MatpowerHandler(network_file)
        stop = time.time()
        # print("Read Matpower: {:g}(s)".format(stop - start))
        self.gen_df_org = self.imat.get_table('gen').copy()
        self.bus_df_org = self.imat.get_table('bus').copy()
        self.gids = self._assign_gen_ids(self.gen_df_org)

        # print("Reading in load data...")
        # Read in the load dataframes
        start = time.time()
        if real_load_file is None:
            self.p_load_df = None
            # raise ValueError("The real load file has not been specified.")
        else:
            p_df = pd.read_csv(real_load_file, index_col=0, parse_dates=True)
            p_df.index = p_df.index.map(lambda t: t.replace(year=year))
            self.p_load_df = p_df

        if reactive_load_file is None:
            self.q_load_df = None
            # raise ValueError("The reactive load file has not been specified.")
        else:
            q_df = pd.read_csv(reactive_load_file, index_col=0, parse_dates=True)
            q_df.index = q_df.index.map(lambda t: t.replace(year=year))
            self.q_load_df = q_df
        stop = time.time()

        gen_type = self.imat.static['genfuel'].split('\n')[1:-2]
        assert len(gen_type) == self.gen_df_org.shape[0]
        self.gen_type = pd.Series(map(lambda s: s.strip(' \t\r\n\';)'), gen_type))

        idx = self._wind_gens(self.gen_type)

        buses = self.gen_df_org.loc[idx,'bus']
        gen_id = pd.Series(data=1, index=buses.index)
        for bus in set(buses):
            bidx = buses == bus
            gen_id.loc[bidx] = range(1,sum(bidx)+1)

        pmax = self.gen_df_org.loc[idx,'Pmax']
        pmax.index = (buses.apply(str)
                      + '_Wind_'
                      + gen_id.apply(str))
        self.wind_max = pmax

        # Recover ExaGO executables, we will check if the exist in PATH and
        # EXAGO_INSTALL_DIR
        self.opflow_executable = self._check_for_exago('opflow')
        self.sopflow_executable = self._check_for_exago('sopflow')
        print("opflow executable = ", self.opflow_executable)
        print("sopflow executable = ", self.sopflow_executable)

        stop_init = time.time()
        # print("Init complete. Time: {:g}(s)".format(stop_init - start_init))

        return

    def _check_for_exago(self, executable_name):
        # This function checks if an exago executable exists
        # Step 1: Check for exago executable in PATH
        val = 0
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, executable_name)
            if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                executable_full_path = path + '/' + executable_name
                val += 1
                print("ExaGO executable {0} found in PATH".format(executable_name))
                return executable_full_path

        assert val == 0
        print("ExaGO executables not found in PATH, checking in EXAGO_INSTALL_DIR")
        if "EXAGO_INSTALL_DIR" in os.environ:
            exe_file = os.path.join(os.environ["EXAGO_INSTALL_DIR"], 'sopflow')
            if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                # print("here2!")
                executable_full_path = os.environ["EXAGO_INSTALL_DIR"] + '/' + executable_name
                val += 1
                print("ExaGO executable {0} found in EXAGO_INSTALL_DIR".format(executable_name))
                return executable_full_path
        else:
            raise ValueError("ExaGO executables not found either in $PATH or $EXAGO_INSTALL_DIR. Please use the former to point to the executables")


    def _non_wind_gens(self,
                       gen_type):
        return np.logical_not(self._wind_gens(gen_type))

    def _wind_gens(self,
                   gen_type):
        return (gen_type == 'wind')

    def _restore_org_gen_table(self):
        gen_df = self.imat.get_table('gen')
        for col in ['Pmax', 'Pmin', 'Pg', 'Qmax', 'Qmin', 'Qg']:
            gen_df.loc[:,col] = self.gen_df_org.loc[:,col]
        return

    def _set_load(self,
                  p_df,
                  q_df,
                  bus_df):
        # Bus loads not in the given dataframes are assumed to be zero
        bus_df.loc[:,'Pd'] = 0.0
        for bus in pd.to_numeric(p_df.columns):
            idx = (bus_df.loc[:,'bus_i'] == bus)
            bus_df.loc[idx,'Pd'] = p_df.loc[p_df.index[0],str(bus)]
        bus_df.loc[:,'Qd'] = 0.0
        for bus in pd.to_numeric(q_df.columns):
            idx = (bus_df.loc[:,'bus_i'] == bus)
            bus_df.loc[idx,'Qd'] = q_df.loc[q_df.index[0],str(bus)]
        return

    def _scale_load(self, bus_df, scaling_factor):
        bus_df.loc[:,'Pd'] = scaling_factor*bus_df.loc[:,'Pd']
        bus_df.loc[:,'Qd'] = scaling_factor*bus_df.loc[:,'Qd']
        return

    def _set_wind(self,
                  w_df,
                  gen_df,
                  gen_type):
        idx = self._wind_gens(gen_type)
        assert w_df.columns.size == idx.sum()
        gen_df.loc[idx,'Pmin'] = 0.0

        for wgen in w_df.columns:
            bus = int(wgen.split('_')[0])
            gen_id = int(wgen.split('_')[2]) - 1
            widx = (gen_df.loc[:,'bus'] == bus) & (idx)
            if widx.sum() > 1:
                # print("Unable to uniquely identify row corresponding to generator {}".format(wgen))
                # print(gen_df.loc[widx,:])

                # Assume that the gen_id gives the row number of whatever is
                # left. Should be true if MATPOWER and AUX files have generators
                # listed in the same order.
                gen_df.loc[widx,'Pmax'].iloc[gen_id] = w_df.loc[w_df.index[0],
                                                                wgen]
            elif widx.sum() == 0:
                print("Unable to identify row corresponding to generator {}".format(wgen))
            else:
                gen_df.loc[widx,'Pmax'] = w_df.loc[w_df.index[0], wgen]

        return

    def _parse_cost(self, matpower_file):
        obj_re_float = re.compile('mpc\.obj = (\d+\.\d+);')
        obj_re_exp = re.compile('mpc\.obj = (\d+\.\d+\w\+\d+);')
        obj_re_int = re.compile('mpc\.obj = (\d+);')
        cost = -1.0

        with open(matpower_file) as mp_file:
            line = mp_file.readline()
            while line:
                m_float = obj_re_float.match(line)
                m_exp = obj_re_exp.match(line)
                m_int = obj_re_int.match(line)
                if m_float:
                    cost = float(m_float.group(1))
                    return cost
                elif m_exp:
                    cost = float(m_exp.group(1))
                    return cost
                elif m_int:
                    warnings.warn("Parsed objective is an Integer. Check solve to ensure everything okay.")
                    cost = float(m_int.group(1))
                    return cost
                line = mp_file.readline()

        return cost

    def _cleanup(self):
        if path.exists("opflowout.m"):
            os.remove("opflowout.m")
        if path.exists("sopflowout"):
            shutil.rmtree("sopflowout")
        if path.exists("case_{0}.m".format(self.grid_name)):
            os.remove("case_{0}.m".format(self.grid_name))
        if path.exists("case_{0}.m".format(self.grid_name)):
            os.remove("scen_{0}.m".format(self.grid_name))

    def _assign_gen_ids(self, gen_df):
        gbuses = gen_df.loc[:,'bus']
        gids = ['1 '] * gbuses.size
        if gbuses.size > gbuses.unique().size:
            for (k,bus) in enumerate(gbuses.unique()):
                idx = gen_df.loc[:,'bus'] == bus
                if idx.sum() > 1:
                    for (gid,gbus) in enumerate(gen_df.loc[idx,'bus']):
                        print("Giving generator at bus {} id {}".format(gbus, gid+1))
                        assert bus == gbus
                        gids[k] = "{:<2d}".format(gid)
                else:
                    pass
        elif gbuses.size < gbuses.unique().size:
            assert False
        else:
            pass
        return gids

    def base_cost(self,
                  start_time,
                  pv_fcst_df, # Currently unused
                  wind_fcst_df,
                  prev_set_points, # Currently unused
                  n_periods, # Currently unused
                  step, # Currently unused
                  opflow_options_dict,
                  system="Summit"
                  ):

        t0 = time.time()

        stop_time = start_time
        self._restore_org_gen_table()

        ## # Uncomment for custom load
        # p_df = self.p_load_df.loc[start_time:stop_time, :]
        # q_df = self.q_load_df.loc[start_time:stop_time, :]
        # self._set_load(p_df, q_df, self.imat.get_table('bus'))
        # self._scale_load(self.imat.get_table('bus'), 0.9)

        wf_df = wind_fcst_df
        # display(wf_df)
        print("start_time = ", start_time)
        self._set_wind(wf_df, self.imat.get_table('gen'), self.gen_type)

        t1 = time.time()

        matpower_file = os.path.join('case_{}.m'.format(self.grid_name))
        self.imat.write_matpower_file(matpower_file)

        t2 = time.time()

        if system == "Summit":
            # For Summit
            exago_cmd = [opflow_options_dict['job_launcher'],
                         '-n', opflow_options_dict['n_mpi_procs'],
                         '-a', '1',
                         '-c', '1',
                         '-g', '0',
                         self.opflow_executable,
                         '-netfile',
                         matpower_file,
                         '-opflow_solver',
                         opflow_options_dict['opflow_solver'], # 'IPOPT',
                         '-save_output',
                         '-opflow_initialization',
                         'ACPF',
                         ]
        else:
            # exago_cmd = [opflow_options_dict['job_launcher'],
            #              '-n',
            #              '1',
            #              self.opflow_executable,
            #              '-netfile',
            #              matpower_file,
            #              '-opflow_solver',
            #              opflow_options_dict['opflow_solver'], # 'IPOPT',
            #              '-save_output',
            #              '-opflow_initialization',
            #              'ACPF',
            #              ]
            options_file = os.path.join(self.options_dir, "opflowoptions")
            print("options_file: {}".format(options_file))

            exago_cmd = [opflow_options_dict['job_launcher'],
                         '-n', '1',
                         self.opflow_executable,
                         '-options_file', options_file,
                         '-save_output',
                         ]


        print("**** ExaGO Command ****\n", exago_cmd, flush=True)
        exago_res = subprocess.run(exago_cmd, capture_output=True, text=True)
        # print(exago_res.stdout)
        if exago_res.stderr:
            print(exago_res.stderr)

        t3 = time.time()

        print("output_path: {}".format(opflow_options_dict['output_path']))

        result = MatpowerHandler(opflow_options_dict['output_path']) # Not properly utilized because of ExaGO

        set_points = result.get_table('gen')

        t4 = time.time()

        obj = self._parse_cost('opflowout.m')

        t5 = time.time()

        elapsed = t5 - t0
        print("""**** Base Cost Timing ****
Change Tables: {:g}(s)  {:g}(%)
Write Tables: {:g}(s)  {:g}(%)
ExaGO: {:g}(s)  {:g}(%)
Set Points: {:g}(s)  {:g}(%)
Base Cost: {:g}(s)  {:g}(%)
Total: {:g}(s)
""".format(
    t1 - t0, (t1 - t0)/elapsed * 100,
    t2 - t1, (t2 - t1)/elapsed * 100,
    t3 - t2, (t3 - t2)/elapsed * 100,
    t4 - t3, (t4 - t3)/elapsed * 100,
    t5 - t4, (t5 - t4)/elapsed * 100,
    elapsed
),
              flush=True
              )

        return (obj, set_points)
