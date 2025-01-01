import os
import time
import numpy as np
import re
from collections import defaultdict
import copy

class GalvatronProfiler():
    def __init__(self, args):
        self.args = args
        self.layernum_arg_names = None
        self.mem_path = None
        self.time_path = None
        self.model_name = None

    def set_path(self, path):
        self.path = path
