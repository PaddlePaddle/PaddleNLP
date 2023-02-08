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

import distutils.util
import logging
import os
import random
from dataclasses import dataclass, field

import h5py
import numpy as np
import paddle
from paddle.io import Dataset

from paddlenlp.data import Stack
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    BertForPretraining,
    BertModel,
    BertPretrainingCriterion,
    BertTokenizer,
    ErnieForPretraining,
    ErnieModel,
    ErniePretrainingCriterion,
    ErnieTokenizer,
    LinearDecayWithWarmup,
)

FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertModel, BertForPretraining, BertPretrainingCriterion, BertTokenizer),
    "ernie": (ErnieModel, ErnieForPretraining, ErniePretrainingCriterion, ErnieTokenizer),
}


@dataclass
class DataArguments:
    input_dir: str = field(default=None, metadata={"help": "The input directory where the data will be read from."})


@dataclass
class ModelArguments:
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(
                sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])
            )
        },
    )
    max_predictions_per_seq: int = field(
        default=80, metadata={"help": "The maximum total of masked tokens in input sequence"}
    )

    to_static: distutils.util.strtobool = field(default=False, metadata={"help": "Enable training under @to_static."})
    profiler_options: str = field(
        default=None,
        metadata={"help": "Whether to use FusedTransformerEncoderLayer to replace a TransformerEncoderLayer or not."},
    )
    fuse_transformer: distutils.util.strtobool = field(
        default=False,
        metadata={"help": "Whether to use FusedTransformerEncoderLayer to replace a TransformerEncoderLayer or not."},
    )


def get_train_data_file(data_args):
    files = [
        os.path.join(data_args.input_dir, f)
        for f in os.listdir(data_args.input_dir)
        if os.path.isfile(os.path.join(data_args.input_dir, f)) and "train" in f
    ]
    files.sort()
    num_files = len(files)
    # random.Random(training_args.seed + epoch).shuffle(files)
    f_start_id = 0

    if paddle.distributed.get_world_size() > num_files:
        remainder = paddle.distributed.get_world_size() % num_files
        data_file = files[
            (f_start_id * paddle.distributed.get_world_size() + paddle.distributed.get_rank() + remainder * f_start_id)
            % num_files
        ]
    else:
        data_file = files[
            (f_start_id * paddle.distributed.get_world_size() + paddle.distributed.get_rank()) % num_files
        ]

    # TODO(guosheng): better way to process single file
    single_file = True if f_start_id + 1 == len(files) else False

    for f_id in range(f_start_id, len(files)):
        if not single_file and f_id == f_start_id:
            continue
        if paddle.distributed.get_world_size() > num_files:
            data_file = files[
                (f_id * paddle.distributed.get_world_size() + paddle.distributed.get_rank() + remainder * f_id)
                % num_files
            ]
        else:
            data_file = files[(f_id * paddle.distributed.get_world_size() + paddle.distributed.get_rank()) % num_files]

    return data_file


def set_seed(seed):
    random.seed(seed + paddle.distributed.get_rank())
    np.random.seed(seed + paddle.distributed.get_rank())
    paddle.seed(seed + paddle.distributed.get_rank())


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def data_collator(data, stack_fn=Stack()):
    num_fields = len(data[0])
    out = [None] * num_fields
    # input_ids, segment_ids, input_mask, masked_lm_positions,
    # masked_lm_labels, next_sentence_labels, mask_token_num
    for i in (0, 1, 2, 5):
        out[i] = stack_fn([x[i] for x in data])
    _, seq_length = out[0].shape
    size = _ = sum(len(x[3]) for x in data)
    # Padding for divisibility by 8 for fp16 or int8 usage
    if size % 8 != 0:
        size += 8 - (size % 8)
    # masked_lm_positions
    # Organize as a 1D tensor for gather or use gather_nd
    out[3] = np.full(size, 0, dtype=np.int32)
    # masked_lm_labels
    out[4] = np.full([size, 1], -1, dtype=np.int64)
    mask_token_num = 0
    for i, x in enumerate(data):
        for j, pos in enumerate(x[3]):
            out[3][mask_token_num] = i * seq_length + pos
            out[4][mask_token_num] = x[4][j]
            mask_token_num += 1
    # mask_token_num
    out.append(np.asarray([mask_token_num], dtype=np.float32))
    return {
        "input_ids": out[0],
        "token_type_ids": out[1],
        "attention_mask": out[2],
        "masked_positions": out[3],
        "labels": (out[4], out[5]),
        "masked_lm_scale": out[6],
    }


def create_input_specs():
    input_ids = paddle.static.InputSpec(name="input_ids", shape=[-1, -1], dtype="int64")
    segment_ids = paddle.static.InputSpec(name="segment_ids", shape=[-1, -1], dtype="int64")
    position_ids = None
    input_mask = paddle.static.InputSpec(name="input_mask", shape=[-1, 1, 1, -1], dtype="float32")
    masked_lm_positions = paddle.static.InputSpec(name="masked_lm_positions", shape=[-1], dtype="int32")
    return [input_ids, segment_ids, position_ids, input_mask, masked_lm_positions]


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            input[index].astype(np.int64) if indice < 5 else np.asarray(input[index].astype(np.int64))
            for indice, input in enumerate(self.inputs)
        ]
        # TODO: whether to use reversed mask by changing 1s and 0s to be
        # consistent with nv bert
        input_mask = (1 - np.reshape(input_mask.astype(np.float32), [1, 1, input_mask.shape[0]])) * -1e9

        index = self.max_pred_length
        # store number of  masked tokens in index
        # outputs of torch.nonzero diff with that of numpy.nonzero by zip
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        else:
            index = self.max_pred_length
        # masked_lm_labels = np.full(input_ids.shape, -1, dtype=np.int64)
        # masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)

        return [input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_labels, next_sentence_labels]


class PretrainingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        elif self.criterion is not None and "start_positions" in inputs and "end_positions" in inputs:
            labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
        elif self.criterion is not None and "generator_labels" in inputs:
            labels = inputs["generator_labels"]
        else:
            labels = None
        # TODO: label_names pop

        masked_lm_scale = inputs.pop("masked_lm_scale")
        outputs = model(**inputs)

        if self.criterion is not None:
            loss = self.criterion(outputs, labels, masked_lm_scale)
            outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def do_train():
    data_args, training_args, model_args = PdArgumentParser(
        [DataArguments, TrainingArguments, ModelArguments]
    ).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args
    data_args: DataArguments = data_args

    training_args.print_config(data_args, "Data")
    training_args.print_config(model_args, "Model")
    training_args.print_config(model_args, "Training")

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    model_args.model_type = model_args.model_type.lower()
    base_class, model_class, criterion_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

    pretrained_models_list = list(model_class.pretrained_init_configuration.keys())
    pretrained_models_list.append("__internal_testing__/bert")

    if model_args.model_name_or_path in pretrained_models_list:
        config = model_class.config_class.from_pretrained(model_args.model_name_or_path)
        config.fuse = model_args.fuse_transformer
        model = model_class(config)
    else:
        model = model_class.from_pretrained(model_args.model_name_or_path)

    data_file = get_train_data_file(data_args)
    train_dataset = PretrainingDataset(input_file=data_file, max_pred_length=model_args.max_predictions_per_seq)

    class CriterionWrapper(paddle.nn.Layer):
        def __init__(self):
            """CriterionWrapper"""
            super(CriterionWrapper, self).__init__()
            self.criterion = criterion_class(getattr(model, model_class.base_model_prefix).config.vocab_size)

        def forward(self, output, labels, masked_lm_scale):
            # input_ids, segment_ids, input_mask, masked_lm_positions,
            # masked_lm_labels, next_sentence_labels, mask_token_num
            masked_lm_labels, next_sentence_labels = labels
            prediction_scores, seq_relationship_score = output
            loss = self.criterion(
                prediction_scores,
                seq_relationship_score,
                masked_lm_labels,
                next_sentence_labels,
                masked_lm_scale,
            )
            return loss

    # decorate @to_static for benchmark, skip it by default.
    if model_args.to_static:
        specs = create_input_specs()
        model = paddle.jit.to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(specs))

    # If use default last_epoch, lr of the first iteration is 0.
    # Use `last_epoch = 0` to be consistent with nv bert.
    num_training_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
    )

    lr_scheduler = LinearDecayWithWarmup(
        training_args.learning_rate, num_training_steps, training_args.warmup_steps, last_epoch=0
    )

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=training_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if training_args.fp16:
        model = paddle.amp.decorate(models=model, level=training_args.fp16_opt_level, save_dtype="float32")

    trainer = PretrainingTrainer(
        model=model,
        criterion=CriterionWrapper(),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        optimizers=[optimizer, lr_scheduler],
    )
    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    do_train()
