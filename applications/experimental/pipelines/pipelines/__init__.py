# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = '0.1.0a0'  # Maybe dev is better

from typing import Union
from types import ModuleType

try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata  # type: ignore

# This configuration must be done before any import to apply to all submodules
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.WARNING)
logging.getLogger("pipelines").setLevel(logging.INFO)

from pipelines import utils
from pipelines import pipelines
from pipelines.schema import Document, Answer, Label, Span
from pipelines.nodes import BaseComponent
from pipelines.pipelines import Pipeline

import pandas as pd

pd.options.display.max_colwidth = 80

# ###########################################
# Enable old style imports (temporary)
import sys

logger = logging.getLogger(__name__)


# Wrapper emitting a warning on import
def DeprecatedModule(mod,
                     deprecated_attributes=None,
                     is_module_deprecated=True):
    """
    Return a wrapped object that warns about deprecated accesses at import
    """

    class DeprecationWrapper:
        warned = []

        def __getattr__(self, attr):
            is_a_deprecated_attr = deprecated_attributes and attr in deprecated_attributes
            is_a_deprecated_module = is_module_deprecated and attr not in [
                "__path__", "__spec__", "__name__"
            ]
            warning_already_emitted = attr in self.warned
            attribute_exists = getattr(mod, attr) is not None

            if (is_a_deprecated_attr or is_a_deprecated_module
                ) and not warning_already_emitted and attribute_exists:
                logger.warn(
                    f"Object '{attr}' is imported through a deprecated path. Please check out the docs for the new import path."
                )
                self.warned.append(attr)
            return getattr(mod, attr)

    return DeprecationWrapper()


# All modules to be aliased need to be imported here

# This self-import is used to monkey-patch, keep for now
import pipelines  # pylint: disable=import-self
from pipelines.nodes import (file_converter, preprocessor, ranker, reader,
                             retriever)

# Note that we ignore the ImportError here because if the user did not install
# the correct dependency group for a document store, we don't need to setup
# import warnings for that, so the import here is useless and should fail silently.

document_stores: Union[ModuleType, None] = None
try:
    from pipelines import document_stores
except ImportError:
    pass

from pipelines.nodes.file_classifier import FileTypeClassifier
from pipelines.nodes.other import JoinDocuments, Docs2Answers, JoinAnswers
from pipelines.utils import preprocessing
from pipelines.utils import cleaning

# For the alias to work as an importable module (like `from pipelines import reader`),
# modules need to be set as attributes of their parent model.
# To make chain imports work (`from pipelines.reader import FARMReader`) the module
# needs to be also present in sys.modules with its complete import path.

setattr(preprocessor, "utils", DeprecatedModule(preprocessing))
setattr(preprocessor, "cleaning", DeprecatedModule(cleaning))
sys.modules["pipelines.preprocessor.utils"] = DeprecatedModule(preprocessing)
sys.modules["pipelines.preprocessor.cleaning"] = DeprecatedModule(cleaning)

setattr(pipelines, "document_store", DeprecatedModule(document_stores))
setattr(
    pipelines, "file_converter",
    DeprecatedModule(file_converter,
                     deprecated_attributes=["FileTypeClassifier"]))
setattr(
    pipelines,
    "pipeline",
    DeprecatedModule(
        pipelines,
        deprecated_attributes=[
            "JoinDocuments",
            "Docs2Answers",
        ],
    ),
)
setattr(
    pipelines, "preprocessor",
    DeprecatedModule(preprocessor, deprecated_attributes=["utils", "cleaning"]))
setattr(pipelines, "ranker", DeprecatedModule(ranker))
setattr(pipelines, "reader", DeprecatedModule(reader))
setattr(pipelines, "retriever", DeprecatedModule(retriever))

sys.modules["pipelines.document_store"] = DeprecatedModule(document_stores)
sys.modules["pipelines.file_converter"] = DeprecatedModule(file_converter)
sys.modules["pipelines.pipeline"] = DeprecatedModule(pipelines)
sys.modules["pipelines.preprocessor"] = DeprecatedModule(
    preprocessor, deprecated_attributes=["utils", "cleaning"])
sys.modules["pipelines.ranker"] = DeprecatedModule(ranker)
sys.modules["pipelines.reader"] = DeprecatedModule(reader)
sys.modules["pipelines.retriever"] = DeprecatedModule(retriever)

# To be imported from modules, classes need only to be set as attributes,
# they don't need to be present in sys.modules too.
# Adding them to sys.modules would enable `import pipelines.pipelines.JoinDocuments`,
# which I believe it's a very rare import style.
setattr(file_converter, "FileTypeClassifier", FileTypeClassifier)
setattr(pipelines, "Docs2Answers", Docs2Answers)

# This last line is used to throw the deprecation error for imports like `from pipelines import connector`
deprecated_attributes = [
    "document_store",
    "file_converter",
    "pipeline",
    "preprocessor",
    "ranker",
    "reader",
    "retriever",
]

sys.modules["pipelines"] = DeprecatedModule(
    pipelines,
    is_module_deprecated=False,
    deprecated_attributes=deprecated_attributes)
