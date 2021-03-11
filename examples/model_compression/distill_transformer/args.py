# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import argparse

from data import MODEL_CLASSES, METRIC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )

    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )

    parser.add_argument(
        "--teacher_finetuned_model_path",
        default=None,
        type=str,
        help="Path to teacher finetuned model.")

    parser.add_argument(
        "--pretrained_model_path",
        default="/root/.paddlenlp/models",
        type=str,
        help="Pretrained model's path.")

    parser.add_argument(
        "--num_layers_of_student_model",
        default=6,
        type=int,
        help="The number of transformer encoder layers in student model.", )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
        help="Total number of training epochs to perform.", )

    parser.add_argument(
        "--warmup_proportion",
        default=0.,
        type=float,
        help="Linear warmup proportion over total steps.")

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps.")

    parser.add_argument(
        "--save_steps", type=int, default=10, help="Log every X updates steps.")

    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate, and it could be 5e-5, 2e-5, 1e-5 and so on."
    )

    parser.add_argument(
        "--strategy",
        default="skip",
        type=str,
        help="Stragegy for student model's learning from teacher model.")

    parser.add_argument(
        "--k",
        default=2,
        type=int,
        help="Student model learns teacher model in every k layers(skip strategy) or from the last k layers(last strategy)."
    )

    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help="It balances the importance of ce loss and the distillation loss, and it could be 0.2, 0.5, 0.7 and so on."
    )

    parser.add_argument(
        "--T",
        default=5,
        type=float,
        help="It controls how much to rely on the teacher's soft predictions. It could be 5, 10, 20 and so on."
    )

    parser.add_argument(
        "--beta",
        default=10,
        type=int,
        help="It weights the importance of the features for distillation in the intermediate layers, and it could be 10, 100, 500, 1000 so on."
    )

    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--seed", default=2021, type=int, help="Random seed for initialization")

    args = parser.parse_args()
    return args
