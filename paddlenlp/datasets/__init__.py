# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .chnsenticorp import *
from .dataset import *
from .glue import *
from .imdb import *
from .lcqmc import *
from .msra_ner import *
from .peoples_daily_ner import *
from .ptb import *
from .squad import *
from .translation import *
from .dureader import *
from .poetry import *
from .couplet import *
from .experimental import load_dataset, DatasetBuilder, MapDataset, IterDataset