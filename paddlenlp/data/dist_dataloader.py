# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from paddle.distributed import fleet

from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

_MAX_DATA_DIM = 64


class DummyDataset(paddle.io.Dataset):
    """
    A dummy dataset.
    """

    def __len__(self):
        return 0


class DistDataLoader(paddle.io.DataLoader):
    """
    DistDataLoader is a wrapper of paddle.io.DataLoader.
    """

    def __init__(
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
        prefetch_factor=2,
        use_shared_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,
    ):

        if dataset is None:
            dataset = DummyDataset()
            batch_sampler = DistributedBatchSampler(dataset, 1)
            logger.info("rank has no data, use Dummpy dataset")

        super().__init__(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers)

        self._hcg = fleet.get_hybrid_communicate_group()

        # init pp data comm group
        if self._hcg.get_pipe_parallel_world_size() > 1:
            self._pp_data_group = self._init_dataloader_comm_group()
        else:
            self._pp_data_group = None

        # tensor parallel message
        self.mp_group = self._hcg.get_model_parallel_group()
        self.mp_rank = self._hcg.get_model_parallel_rank()
        self.mp_src_rank = self._hcg.get_model_parallel_group_src_rank()

        self.pp_rank = self._hcg.get_stage_id()
        self.dp_rank = self._hcg.get_data_parallel_rank()
        sharding_rank = self._hcg.get_sharding_parallel_rank()
        self._need_data = (self.mp_rank == 0) and (self.pp_rank == 0)
        self._data_keys, self._data_keys_size = None, None

        if self._need_data:
            self._dataloader = paddle.io.DataLoader(
                dataset,
                feed_list,
                places,
                return_list,
                batch_sampler,
                batch_size,
                shuffle,
                drop_last,
                collate_fn,
                num_workers,
                use_buffer_reader,
                prefetch_factor,
                use_shared_memory,
                timeout,
                worker_init_fn,
                persistent_workers,
            )

            self._lazy_dataloader_iter = None
        else:
            logger.info(
                "mp{}_pp{}_sharding{}_dp{} no data needed, "
                "skip init dataloader.".format(self.mp_rank, self.pp_rank, sharding_rank, self.dp_rank)
            )

    @property
    def _dataloader_iter(self):
        if self._lazy_dataloader_iter is None:
            self._lazy_dataloader_iter = iter(self._dataloader)
        return self._lazy_dataloader_iter

    def __len__(self):
        if self._need_data:
            return super().__len__()
        else:
            raise ValueError("raise error for `paddlenlp.trainer.trainer_utils.has_length`")

    def _init_dataloader_comm_group(self):
        topo = self._hcg._topo
        parallel_comm_group = None
        parallel_groups = topo.get_comm_list("pipe")

        for group in parallel_groups:
            # only first rank and last rank
            ranks = [group[0], group[-1]]
            comm_group = paddle.distributed.new_group(ranks=ranks)
            if paddle.distributed.get_rank() in ranks:
                parallel_comm_group = comm_group
        return parallel_comm_group

    def __iter__(self):
        return self

    def __next__(self):
        data_keys_size = 0
        if self._need_data:
            # {'input_ids': int64, 'labels': int64}
            data = next(self._dataloader_iter)
            data_keys_size, data_keys = len(data.keys()), list(data.keys())
            data_list = [data[key] for key in data_keys]
            # TODO(daisiming): add more type assertion.
            assert {item.dtype for item in data_list} == {
                paddle.int64
            }, f"Distloader requires dtype == `int64`, got:{[item.dtype for item in data_list]}"

        # broadcast data keys size
        data_keys_size = paddle.to_tensor(data_keys_size)
        if self._data_keys_size is None:
            if self.mp_group is not None and self.pp_rank == 0:
                paddle.distributed.broadcast(data_keys_size, src=self.mp_src_rank, group=self.mp_group)
            if self._pp_data_group is not None:
                paddle.distributed.broadcast(
                    data_keys_size, src=self._pp_data_group.ranks[0], group=self._pp_data_group
                )
            self._data_keys_size = int(data_keys_size.item())

        if not self._need_data:
            data_keys = [None for i in range(self._data_keys_size)]

        # broadcast data keys name
        if self._data_keys is None:
            if self.mp_group is not None and self.pp_rank == 0:
                paddle.distributed.broadcast_object_list(data_keys, src=self.mp_src_rank, group=self.mp_group)
            if self._pp_data_group is not None:
                paddle.distributed.broadcast_object_list(
                    data_keys, src=self._pp_data_group.ranks[0], group=self._pp_data_group
                )
            self._data_keys = data_keys

        # broadcast data
        if not self._need_data:
            data_list = [None for i in range(self._data_keys_size)]
        if self.mp_group is not None and self.pp_rank == 0:
            data_list = broadcast_data_list(data_list, paddle.int64, self.mp_rank, self.mp_group, self.mp_src_rank)

        if self._pp_data_group is not None:
            # Note(daisimng): In last stage of pp, we don't need input_ids.
            # It will be removed in future.
            data_list = broadcast_data_list(
                data_list,
                paddle.int64,
                self.pp_rank,
                self._pp_data_group,
                self._pp_data_group.ranks[0],
            )

        out = dict([(key, data) for key, data in zip(self._data_keys, data_list)])
        return out


def broadcast_data_list(data_list, datatype, comm_rank=0, comm_group=None, src_rank=0):
    """
    Broadcast data from src_rank to all ranks in comm_group.
    """
    # Move to GPU and broadcast.
    size_cpu = []
    if comm_rank == 0:
        for data in data_list:
            size_cpu.append(len(data.shape))
            size_cpu += data.shape
    size_cpu = size_cpu + [0] * (_MAX_DATA_DIM - len(size_cpu))
    size_cuda = paddle.to_tensor(size_cpu)
    paddle.distributed.broadcast(size_cuda, src_rank, group=comm_group).wait()

    size_cpu = size_cuda.tolist()
    i = 0
    numel = 0
    sizes = []
    while size_cpu[i] > 0:
        rank = size_cpu[i]
        this_size = size_cpu[i + 1 : i + 1 + rank]
        numel += int(np.prod(this_size))
        sizes.append(this_size)
        i += rank + 1

    if comm_rank == 0:
        assert data.dtype == datatype, "input has data type {} which " "is different than {}".format(
            data.dtype, datatype
        )
        data_b = paddle.concat([d.cuda().reshape([-1]) for d in data_list], 0)
        assert numel == sum([d.numel().item() for d in data_list]), (numel, [d.numel().item() for d in data_list])
    else:
        data_b = paddle.empty([numel], dtype=datatype).cuda()

    # Broadcast
    paddle.distributed.broadcast(data_b, src_rank, group=comm_group).wait()

    ret = []
    offset = 0
    for size in sizes:
        numel = int(np.prod(size))
        ret.append(data_b[offset : offset + numel].reshape(size))
        offset += numel

    return ret
