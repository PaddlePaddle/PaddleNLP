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

from .version import VERSION

__version__ = VERSION  # Maybe dev is better
import sys

if "datasets" in sys.modules.keys():
    from paddlenlp.utils.log import logger

    logger.warning(
        "Detected that datasets module was imported before paddlenlp. "
        "This may cause PaddleNLP datasets to be unavalible in intranet. "
        "Please import paddlenlp before datasets module to avoid download issues"
    )
import paddle

from . import (
    data,
    dataaug,
    datasets,
    embeddings,
    experimental,
    layers,
    losses,
    metrics,
    ops,
    prompt,
    seq2vec,
    trainer,
    transformers,
    utils,
)
from .server import SimpleServer
from .taskflow import Taskflow

paddle.disable_signal_handler()

import sys
import warnings

# BatchSampler use `_non_static_mode` which is not included in version <= 2.3,
# thus use `in_dynamic_mode` instead.
from paddle import in_dynamic_mode

# Patches for DataLoader/BatchSamper to allow using other than paddle.io.Dataset
# in Paddle version lower than 2.3
from paddle.fluid.reader import (
    BatchSampler,
    IterableDataset,
    _convert_places,
    _current_expected_place,
    _DatasetKind,
    _get_paddle_place,
    _get_paddle_place_list,
    _InfiniteIterableSampler,
    use_pinned_memory,
)


def _patch_data_loader_init(
    self,
    dataset,
    feed_list=None,
    places=None,
    return_list=True,
    batch_sampler=None,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    collate_fn=None,
    num_workers=0,
    use_buffer_reader=True,
    use_shared_memory=True,
    timeout=0,
    worker_init_fn=None,
    persistent_workers=False,
):
    self.return_list = return_list
    self.collate_fn = collate_fn
    self.use_buffer_reader = use_buffer_reader
    self.worker_init_fn = worker_init_fn

    self.dataset = dataset

    if not return_list and not in_dynamic_mode():
        assert feed_list is not None, "feed_list should be set when return_list=False"
    self.feed_list = feed_list

    if places is None:
        places = _current_expected_place()
    if isinstance(places, (list, tuple)):
        places = _get_paddle_place_list(places)
    else:
        places = _get_paddle_place(places)
    self.places = _convert_places(places)

    assert num_workers >= 0, "num_workers should be a non-negative value"
    if num_workers > 0 and (sys.platform == "darwin" or sys.platform == "win32"):
        warnings.warn(
            "DataLoader with multi-process mode is not supported on MacOs and Windows currently."
            " Please use signle-process mode with num_workers = 0 instead"
        )
        num_workers = 0
    self.num_workers = num_workers

    self.use_shared_memory = use_shared_memory
    if use_shared_memory and num_workers == 0:
        self.use_shared_memory = False

    assert timeout >= 0, "timeout should be a non-negative value"
    self.timeout = timeout

    if isinstance(dataset, IterableDataset):
        self.dataset_kind = _DatasetKind.ITER
        if shuffle:
            raise ValueError("IterableDataset not support shuffle, but got shuffle={}".format(shuffle))
        if batch_sampler is not None:
            raise ValueError("IterableDataset expect unspecified batch_sampler")
    else:
        self.dataset_kind = _DatasetKind.MAP

    if batch_sampler is not None:
        assert batch_size == 1 and not shuffle and not drop_last, (
            "batch_size/shuffle/drop_last should not be set when " "batch_sampler is given"
        )
        self.batch_sampler = batch_sampler
        self.batch_size = None
    elif batch_size is None:
        self.batch_sampler = None
        self.batch_size = None
    else:
        assert batch_size > 0, "batch_size should be None or a positive value when " "batch_sampler is not given"
        self.batch_size = batch_size
        if isinstance(dataset, IterableDataset):
            self.batch_sampler = _InfiniteIterableSampler(dataset, batch_size)
        else:
            self.batch_sampler = BatchSampler(
                dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
            )

    self.drop_last = drop_last
    self.auto_collate_batch = self.batch_sampler is not None

    self.pin_memory = False
    if in_dynamic_mode():
        self.pin_memory = True if use_pinned_memory() is None else use_pinned_memory()

    self._persistent_workers = persistent_workers
    self._iterator = None


from paddle.fluid.dataloader.batch_sampler import (
    RandomSampler,
    Sampler,
    SequenceSampler,
)


def _patch_batch_sampler_init(self, dataset=None, sampler=None, shuffle=False, batch_size=1, drop_last=False):
    if dataset is None:
        assert sampler is not None, "either dataset or sampler should be set"
        assert isinstance(sampler, Sampler), "sampler should be a paddle.io.Sampler, but got {}".format(type(sampler))
        assert not shuffle, "shuffle should be False when sampler is set"
        self.sampler = sampler
    else:
        assert not isinstance(dataset, IterableDataset), "dataset should not be a paddle.io.IterableDataset"
        assert sampler is None, "should not set both dataset and sampler"
        assert isinstance(shuffle, bool), "shuffle should be a boolean value, but got {}".format(type(shuffle))
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequenceSampler(dataset)

    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), "batch_size should be a positive integer, but got {}".format(batch_size)
    self.batch_size = batch_size
    assert isinstance(drop_last, bool), "drop_last should be a boolean value, but got {}".format(type(drop_last))
    self.drop_last = drop_last


import functools

# Any '2.3.X' version would be bigger than '2.3'
if paddle.__version__ != "0.0.0" and paddle.__version__ < "2.3":
    paddle.io.DataLoader.__init__ = functools.wraps(paddle.io.DataLoader.__init__)(_patch_data_loader_init)
    paddle.io.BatchSampler.__init__ = functools.wraps(paddle.io.BatchSampler.__init__)(_patch_batch_sampler_init)
