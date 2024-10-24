# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import torch
from reprod_log import ReprodDiffHelper, ReprodLogger
from torch.optim import AdamW
from transformers.optimization import get_scheduler as get_hf_scheduler

# define paddle scheduler
from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)

scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "polynomial": PolyDecayWithWarmup,
}


def get_paddle_scheduler(
    learning_rate,
    scheduler_type,
    num_warmup_steps=None,
    num_training_steps=None,
    **scheduler_kwargs,
):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError("requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError("requires `num_training_steps`, please provide that argument.")

    return scheduler_type2cls[scheduler_type](
        learning_rate=learning_rate,
        total_steps=num_training_steps,
        warmup=num_warmup_steps,
        **scheduler_kwargs,
    )


def test_lr():
    diff_helper = ReprodDiffHelper()
    pd_reprod_logger = ReprodLogger()
    hf_reprod_logger = ReprodLogger()
    lr = 3e-5
    num_warmup_steps = 345
    num_training_steps = 1024
    milestone = [100, 300, 500, 700, 900]
    for scheduler_type in ["linear", "cosine", "polynomial"]:
        torch_optimizer = AdamW(torch.nn.Linear(1, 1).parameters(), lr=lr)
        hf_scheduler = get_hf_scheduler(
            name=scheduler_type,
            optimizer=torch_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        pd_scheduler = get_paddle_scheduler(
            learning_rate=lr,
            scheduler_type=scheduler_type,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        for i in range(num_training_steps):
            hf_scheduler.step()
            pd_scheduler.step()
            if i in milestone:
                hf_reprod_logger.add(
                    f"step_{i}_{scheduler_type}_lr",
                    np.array([hf_scheduler.get_last_lr()[-1]]),
                )
                pd_reprod_logger.add(f"step_{i}_{scheduler_type}_lr", np.array([pd_scheduler.get_lr()]))

    diff_helper.compare_info(hf_reprod_logger.data, pd_reprod_logger.data)
    diff_helper.report()


if __name__ == "__main__":
    test_lr()
