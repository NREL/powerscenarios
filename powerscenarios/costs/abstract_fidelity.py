# abstract_fidelity.py
# The following file will contain classes pertaining to different models used
# for costing scenarios. Please this inherit this abstract class when creating
# classes that define the various fidelitities used for costing scenarios
import numpy as np
import pandas as pd
import os, sys, time

class AbstractCostingFidelity(object):
    def __init__(self):
        pass
