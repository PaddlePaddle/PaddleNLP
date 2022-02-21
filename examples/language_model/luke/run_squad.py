# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import argparse
from utils.squad_get_predictions import write_predictions
from utils.squad_postprocess import SQuad_postprocess
from utils.squad_processor import load_examples
from paddlenlp.transformers import LukeTokenizer
from paddlenlp.transformers import LukeForQuestionAnswering
import os
from utils.trainer import Trainer
import paddle
import collections
from tqdm import tqdm

parser = argparse.ArgumentParser(description="LUKE FOR MRC")

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Use to store all outputs during training and evaluation.")
parser.add_argument(
    "--data_dir", type=str, required=True, help="Dataset folder")
parser.add_argument(
    "--do_eval", action='store_true', help="Whether to evaluate the model.")
parser.add_argument(
    "--do_train", action='store_true', help="Whether to train the model.")
parser.add_argument(
    "--num_train_epochs", type=int, default=2, help="Number of training cycles")
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=8,
    help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--device",
    type=str,
    default='gpu',
    help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=3,
    help="Gradient accumulated before each parameter update.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Weight decay if we apply some")
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.06,
    help="Proportion of training steps to perform linear learning rate warmup for."
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=20e-6,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--model_type",
    type=str,
    default='luke-base',
    help="Type of pre-trained model.")
parser.add_argument(
    "--max_answer_length", type=int, default=30, help="Max answer length")
parser.add_argument(
    "--n_best_size", type=int, default=20, help="n-best logits from a list")
parser.add_argument(
    "--wiki_link_db_file",
    type=str,
    default="enwiki_20160305.pkl",
    help="Wikipedia entity dataset")
parser.add_argument(
    "--model_redirects_file",
    type=str,
    default="enwiki_20181220_redirects.pkl",
    help="Wikipedia entity dataset")
parser.add_argument(
    "--link_redirects_file",
    type=str,
    default="enwiki_20160305_redirects.pkl",
    help="Wikipedia entity dataset")
parser.add_argument(
    "--with_negative",
    action='store_true',
    help="Whether to evaluate the model.")
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Max sequence length")
parser.add_argument(
    "--max_mention_length",
    type=int,
    default=30,
    help="Max entity position's length")
parser.add_argument(
    "--doc_stride",
    type=int,
    default=128,
    help="When splitting up a long document into chunks, how much stride to take between chunks."
)
parser.add_argument(
    "--max_query_length", type=int, default=64, help="Max query length.")
parser.add_argument(
    "--min_mention_link_prob",
    type=float,
    default=0.01,
    help="Min mention link prob")

args = parser.parse_args()


@paddle.no_grad()
def evaluate(args, model):
    dataloader, examples, features, processor = load_examples(
        args, data_file='dev-v1.json')
    all_results = []
    logging.info("evaluating the model......")
    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])
    for batch in tqdm(dataloader, desc="eval"):
        model.eval()
        outputs = model(
            input_ids=batch[0],
            token_type_ids=batch[1],
            attention_mask=batch[2],
            entity_ids=batch[3],
            entity_position_ids=batch[4],
            entity_segment_ids=batch[5],
            entity_attention_mask=batch[6])

        for i, example_index in enumerate(batch[-1].reshape((-1, ))):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = [o[i] for o in outputs]
            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            all_results.append(RawResult(unique_id, start_logits, end_logits))
    all_predictions = write_predictions(
        args, examples, features, all_results, args.n_best_size,
        args.max_answer_length, False,
        LukeTokenizer.from_pretrained(args.model_type))
    SQuad_postprocess(
        os.path.join(args.data_dir, processor.dev_file), all_predictions)


if __name__ == '__main__':
    model = LukeForQuestionAnswering.from_pretrained(args.model_type)
    if args.do_train:
        train_dataloader, _, _, _ = load_examples(
            args, data_file='train-v1.json')
        num_train_steps = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs
        trainer = Trainer(
            args,
            model=model,
            dataloader=train_dataloader,
            num_train_steps=num_train_steps)
        trainer.train()
    if args.do_eval:
        model.from_pretrained(args.output_dir)
        evaluate(args, model)
