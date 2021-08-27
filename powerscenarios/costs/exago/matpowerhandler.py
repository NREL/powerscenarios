import numpy as np
import pandas as pd
import os, sys, time
import re
import subprocess
import warnings
from os import path
import shutil

class MatpowerHandler:

    table_cols = {'bus' : ["bus_i",
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
                           ],
                  'gen' : ["bus",
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
                           ],
                  }

    def __init__(self,
                 network_file,
                 dynamic_tables=['bus', 'gen'],
                 static_tables=['branch', 'gencost', 'bus_name', 'gentype', 'genfuel']):

        t0 = time.time()

        self.network_file = network_file

        static_tables = static_tables
        (tables, order) = self._parse_file(network_file,
                                           dynamic_tables + static_tables)

        t1 = time.time()

        self.table_order = order
        (self.header, self.footer, self.dynamic, self.static
         ) = self._split_tables(tables, dynamic_tables, static_tables)
        self._str_to_df(self.dynamic)

        t2 = time.time()

        # print("Parse file: {:g}(s)\nOther: {:g}(s)".format(t1 - t0, t2 - t1))
        self.parse_time = t1 - t0
        self.other_time = t2 - t1

        return

    def _parse_file(self, network_file, tables):

        tables = set(tables + ['header', 'footer'])
        sections = {tab:'' for tab in tables}
        order = []

        table_start = re.compile('^mpc.(\w+) = [\[\{]\s')
        table_end = re.compile('^[\]\}];\s')
        with open(network_file, 'r') as netfile:
            current_table = 'header'
            reading_table = False
            line = netfile.readline()
            while line:
                if reading_table:
                    m = table_end.match(line)
                else:
                    m = table_start.match(line)

                if m:
                    if reading_table:
                        # We have found the end of a table
                        reading_table = False
                        # Add end of table to the string
                        sections[current_table] += line
                        # Assume we will have no more tables until we find one
                        current_table = 'footer'

                    else:
                        # We have found the start of a table
                        reading_table = True
                        # Get which table we're reading
                        current_table = m.group(1)
                        # Add start of table
                        sections[current_table] = line
                        order.append(current_table)
                        # Clear the footer
                        sections['footer'] = ''
                else:
                    # Just the next line in the current table
                    sections[current_table] += line

                line = netfile.readline()

        return (sections, order)

    def _split_tables(self, tables, dynamic, static):
        dynamic_tables={}
        static_tables={}
        for key in tables.keys():
            if key == 'header':
                header = tables[key]
            elif key == 'footer':
                footer = tables[key]
            elif key in dynamic:
                dynamic_tables[key] = tables[key]
            else:
                static_tables[key] = tables[key]
                # if key not in static:
                #     print('Unspecified table "{}" is assumed to be static'.format(key))
        return (header, footer, dynamic_tables, static_tables)

    def _str_to_df(self, dynamic):
        for key in dynamic.keys():
            tab_str = dynamic[key]
            rows = []
            col_names = MatpowerHandler.table_cols[key]
            for line in tab_str.strip().split(sep='\n')[1:-1]:
                vals = line.strip(' \n\r\t;').split(sep='\t')
                rows.append(dict(zip(col_names, vals)))
            df = pd.DataFrame(rows, columns=col_names)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            dynamic[key] = df
        return

    def _write_dynamic_table(self, ostream, table_name, table):
        ostream.write('mpc.{} = [\n'.format(table_name))
        table.to_csv(ostream, sep='\t', index=False, header=False)
        ostream.write('];\n\n')
        return

    def _write_static_table(self, ostream, table_name, table):
        ostream.write(table)
        ostream.write('\n')
        return

    def get_table(self, table_name):
        if table_name in self.dynamic.keys():
            df = self.dynamic[table_name]
        else:
            raise KeyError("Unknown or static table: {}".format(table_name))
        return df

    def write_matpower_file(self, new_file_name):
        with open(new_file_name, 'w') as mp_file:
            self._write_static_table(mp_file, 'header', self.header)
            for tab in self.table_order:
                if tab in self.dynamic.keys():
                    self._write_dynamic_table(mp_file, tab, self.dynamic[tab])
                elif tab in self.static.keys():
                    self._write_static_table(mp_file, tab, self.static[tab])
                else:
                    print('Table {} neither dynamic nor static. Unable to print.'.format(tab))
            self._write_static_table(mp_file, 'footer', self.footer)
        return
