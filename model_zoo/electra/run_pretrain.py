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

import io
import logging
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import paddle

from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    ElectraForTotalPretraining,
    ElectraPretrainingCriterion,
    ElectraTokenizer,
    LinearDecayWithWarmup,
)

FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "electra": (ElectraForTotalPretraining, ElectraTokenizer),
}


@dataclass
class TrainingArguments(TrainingArguments):

    # per_device_train_batch_size
    @property
    def micro_batch_size(self):
        return self.per_device_train_batch_size

    @property
    def eval_freq(self):
        return self.eval_steps


@dataclass
class ModelArguments:
    model_type: str = field(
        default="electra", metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: str = field(
        default="electra-small",
        metadata={
            "help": "Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(
                sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])
            )
        },
    )
    max_seq_length: int = field(default=128, metadata={"help": "max length of each sequence"})
    mask_prob: float = field(default=0.15, metadata={"help": "the probability of one word to be mask"})
    eager_run: bool = field(default=True, metadata={"help": "Use dygraph mode."})
    init_from_ckpt: bool = field(
        default=True,
        metadata={
            "help": "Whether to load model checkpoint. if True, args.model_name_or_path must be dir store ckpt or will train from fresh start"
        },
    )
    max_predictions_per_seq: int = field(
        default=20, metadata={"help": "The maximum total input sequence length after tokenization"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(default=None, metadata={"help": "The input directory where the data will be read from."})
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class BookCorpus(paddle.io.Dataset):
    """
    https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html
    Args:
        data_path (:obj:`str`) : The dataset file path, which contains train.tsv, dev.tsv and test.tsv.
        tokenizer (:obj:`class PretrainedTokenizer`) : The tokenizer to split word and convert word to id.
        max_seq_length (:obj:`int`) : max length for each sequence.
        mode (:obj:`str`, `optional`, defaults to `train`):
            It identifies the dataset mode (train, test or dev).
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_seq_length,
        mode="train",
    ):
        if mode == "train":
            data_file = "train.data"
        elif mode == "test":
            data_file = "test.data"
        else:
            data_file = "dev.data"

        self.data_file = os.path.join(data_path, data_file)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.raw_examples = self._read_file(self.data_file)

    def _read_file(self, input_file):
        """
        Reads a text file.

        Args:
            input_file (:obj:`str`) : The file to be read.

        Returns:
            examples (:obj:`list`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                examples = []
                while True:
                    line = f.readline()
                    if line:
                        if len(line) > 0 and not line.isspace():
                            example = self.tokenizer(line, max_seq_len=self.max_seq_length)["input_ids"]
                            examples.append(example)
                    else:
                        break
                return examples

    def truncation_ids(self, ids, max_seq_length):
        if len(ids) <= (max_seq_length - 2):
            return ids
        else:
            return ids[: (max_seq_length - 2)]

    def __getitem__(self, idx):
        return self.raw_examples[idx]

    def __len__(self):
        return len(self.raw_examples)


class DataCollatorForElectra(object):
    """
    pads, gets batch of tensors and preprocesses batches for masked language modeling
    when dataloader num_worker > 0, this collator may trigger some bugs, for safe, be sure dataloader num_worker=0
    """

    def __init__(self, tokenizer, max_seq_length, mlm=True, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm = True
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        if self.mlm:
            inputs, raw_inputs, labels = self.mask_tokens(examples)
            return {
                "input_ids": inputs,
                "raw_input_ids": raw_inputs,
                "generator_labels": labels,
            }
        else:
            raw_inputs, _ = self.add_special_tokens_and_set_maskprob(examples, True, self.max_seq_length)
            raw_inputs = self.tensorize_batch(raw_inputs, "int64")
            inputs = raw_inputs.clone().detach()
            labels = raw_inputs.clone().detach()
            if self.tokenizer.pad_token is not None:
                pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                labels[labels == pad_token_id] = -100
            return {
                "raw_input_ids": raw_inputs,
                "generator_labels": labels,
            }
            # return batch, raw_inputs, labels  # noqa:821

    def tensorize_batch(self, examples, dtype):
        if isinstance(examples[0], (list, tuple)):
            examples = [paddle.to_tensor(e, dtype=dtype) for e in examples]
        length_of_first = examples[0].shape[0]
        are_tensors_same_length = all(x.shape[0] == length_of_first for x in examples)
        if are_tensors_same_length:
            return paddle.stack(examples, axis=0)
        else:
            raise ValueError("the tensor in examples not have same shape, please check input examples")

    def add_special_tokens_and_set_maskprob(self, inputs, truncation, max_seq_length):
        # sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        full_inputs = []
        full_maskprob = []
        max_length = 0
        for ids in inputs:
            if len(ids) > max_length:
                max_length = len(ids)
        max_length = min(max_length, max_seq_length)

        for ids in inputs:
            if len(ids) <= max_length:
                padding_num = max_length - len(ids)
                full_inputs.append(ids + ([pad_token_id] * padding_num))
                full_maskprob.append([0] + ([self.mlm_probability] * (len(ids) - 2)) + [0] + ([0] * padding_num))
            else:
                if truncation:
                    full_inputs.append(ids[:max_length])
                    full_maskprob.append([0] + ([self.mlm_probability] * (max_length - 2)) + [0])
                else:
                    full_inputs.append(ids)
                    full_maskprob.append([0] + ([self.mlm_probability] * (len(ids) - 2)) + [0])
        return full_inputs, full_maskprob

    def mask_tokens(self, examples):
        if self.tokenizer.mask_token is None:
            raise ValueError("the tokenizer does not have mask_token, please check!")
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        raw_inputs, probability_matrix = self.add_special_tokens_and_set_maskprob(examples, True, self.max_seq_length)
        raw_inputs = self.tensorize_batch(raw_inputs, "int64")
        probability_matrix = self.tensorize_batch(probability_matrix, "float32")
        inputs = raw_inputs.clone()
        labels = raw_inputs.clone()

        total_indices = paddle.bernoulli(probability_matrix).astype("bool").numpy()
        labels[~total_indices] = -100

        # 80% MASK
        indices_mask = paddle.bernoulli(paddle.full(labels.shape, 0.8)).astype("bool").numpy() & total_indices
        inputs[indices_mask] = mask_token_id

        # 10% Random
        indices_random = (
            paddle.bernoulli(paddle.full(labels.shape, 0.5)).astype("bool").numpy() & total_indices & ~indices_mask
        )
        random_words = paddle.randint(low=0, high=self.tokenizer.vocab_size, shape=labels.shape, dtype="int64")
        inputs = paddle.where(paddle.to_tensor(indices_random), random_words, inputs)

        # 10% Original
        return inputs, raw_inputs, labels


class new_Trainer(Trainer):
    def __init__(
        self,
        model=None,
        criterion=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super(new_Trainer, self).__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):

        gen_logits, disc_logits, disc_labels, attention_mask = model(**inputs)
        gen_labels = inputs["generator_labels"]
        loss = self.criterion(gen_logits, disc_logits, gen_labels, disc_labels, attention_mask)
        return loss


def create_dataloader(dataset, mode="train", batch_size=1, use_gpu=True, data_collator=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`):
            Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`):
            If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1):
            The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`True`):
            Whether to use gpu to run.

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """

    if mode == "train" and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True)
        dataloader = paddle.io.DataLoader(
            dataset, batch_sampler=sampler, return_list=True, collate_fn=data_collator, num_workers=0
        )
    else:
        shuffle = True if mode == "train" else False
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = paddle.io.DataLoader(
            dataset, batch_sampler=sampler, return_list=True, collate_fn=data_collator, num_workers=0
        )

    return dataloader


def do_train():
    data_args, training_args, model_args = PdArgumentParser(
        [DataArguments, TrainingArguments, ModelArguments]
    ).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args
    data_args: DataArguments = data_args

    paddle.enable_static() if not model_args.eager_run else None
    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model_args.model_type = model_args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    config = model_class.config_class.from_pretrained(model_args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

    model = model_class(config)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Loads dataset.
    tic_load_data = time.time()
    print("start load data : %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    train_dataset = BookCorpus(
        data_path=data_args.input_dir, tokenizer=tokenizer, max_seq_length=model_args.max_seq_length, mode="train"
    )
    print("load data done, total : %s s" % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    data_collator = DataCollatorForElectra(
        tokenizer=tokenizer, max_seq_length=model_args.max_seq_length, mlm=True, mlm_probability=model_args.mask_prob
    )
    criterion = ElectraPretrainingCriterion(config)

    lr_scheduler = LinearDecayWithWarmup(
        training_args.learning_rate, training_args.max_steps, training_args.warmup_steps
    )

    trainer = new_Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        criterion=criterion,
        optimizers=(None, lr_scheduler),
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
