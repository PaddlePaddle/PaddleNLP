import re


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text

from pycutlass.type_hint import *
from pycutlass.tensor_ref import *
from pycutlass.operation import *
from pycutlass.epilogue import *
from pycutlass.parser import *
from pycutlass.compiler import ArtifactManager
from pycutlass.memory_manager import *
from pycutlass.arguments import *
from pycutlass.library import *
from pycutlass.c_types import *
from pycutlass.gemm_operation import *
from pycutlass.conv2d_operation import *
from pycutlass.compiler import *
from pycutlass.utils import *
from pycutlass.frontend import *
from pycutlass.reduction_operation import *
from pycutlass.compiler import *
from pycutlass.utils.device import device_cc

# module-wide variables

import sys
this = sys.modules[__name__]

# artifact manager
this.compiler = ArtifactManager()

try:
    if not hasattr(this, 'DEVICE_CC') or this.DEVICE_CC is None:
        this.DEVICE_CC = device_cc()
except:
    this.DEVICE_CC = None

def get_memory_pool(init_pool_size=0, max_pool_size=2**34):
    this.memory_pool = PoolMemoryManager(
        init_pool_size=init_pool_size,
        max_pool_size=max_pool_size
    )
    return this.memory_pool
