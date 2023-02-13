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

from dataclasses import dataclass, field

from ..trainer import TrainingArguments
from ..utils.log import logger

__all__ = ["PromptTuningArguments"]


@dataclass
class PromptTuningArguments(TrainingArguments):
    """
    The arguments' subset for training loop during prompt tuning.
    """

    max_seq_length: int = field(default=512, metadata={"help": "The maximum length of all input text."})
    freeze_plm: bool = field(
        default=False, metadata={"help": "If True, the pretrained parameters won't be " "updated during tuning."}
    )
    freeze_dropout: bool = field(
        default=False,
        metadata={
            "help": "If True, pretrained parameters won't be updated " "during tuning and the dropout is disabled."
        },
    )
    save_plm: bool = field(default=False, metadata={"help": "Whether to save pretrained model."})
    use_rdrop: bool = field(
        default=False,
        metadata={
            "help": "Use R-Drop regularization strategy."
            "Please refer to the paper for more details: "
            "https://arxiv.org/abs/2106.14448."
        },
    )
    alpha_rdrop: float = field(default=5.0, metadata={"help": "The KL-divergence loss weight alpha in R-Drop."})
    use_rgl: bool = field(
        default=False,
        metadata={
            "help": "Use label consistency to boost tuning performance."
            "Please refer to the paper for more details: "
            "https://aclanthology.org/2022.findings-naacl.81/."
        },
    )
    alpha_rgl: float = field(default=0.5, metadata={"help": "The weight of label consistency loss in RGL."})

    ppt_learning_rate: float = field(
        default=1e-4, metadata={"help": "The initial learning rate of prompt parameters."}
    )
    ppt_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for the AdamW optimizer of prompt."})
    ppt_adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for the AdamW optimizer of prompt."})
    ppt_adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for the AdamW optimizer of prompt."})
    ppt_adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for the AdamW optimizer of prompt."})

    def __post_init__(self):
        super(PromptTuningArguments, self).__post_init__()
        if self.use_rgl and self.alpha_rgl == 0.0:
            logger.warning(
                "Ignore `use_rgl` because `alpha_rgl` = 0. Please " "set `alpha_rgl` a positive float to use RGL loss."
            )
            self.use_rgl = False

        if self.use_rdrop and self.alpha_rdrop == 0.0:
            logger.warning(
                "Ignore `use_rdrop` because `alpha_rdrop` = 0. Please "
                "set `alpha_rdrop` a positive float to use R-Drop."
            )
            self.use_rdrop = False

        if self.freeze_dropout:
            self.freeze_plm = True
