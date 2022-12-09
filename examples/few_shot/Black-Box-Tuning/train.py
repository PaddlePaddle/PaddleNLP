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

# import fitlog
import argparse
import copy
import os
import random
import time
from functools import partial

import cma
import numpy as np
import paddle
from data import convert_example, create_dataloader, load_data, transform_data
from model import p_ErnieForMaskedLM
from sklearn.metrics import f1_score
from utils import file_path, label2text

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="ernie-3.0-medium-zh", type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--budget", default=20000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default="gpu:1", type=str)
parser.add_argument("--alg", default="CMA", type=str)
parser.add_argument("--random_proj", default="normal", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default="ce", type=str)
parser.add_argument("--split", default=0.2, type=float)
parser.add_argument("--parallel", action="store_true", help="Whether to allow parallel evaluation")

args = parser.parse_args()

# below are free hyper-params
model_name = "ernie-3.0-medium-zh"

n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
bound = args.bound
sigma = args.sigma
max_seq_length = 128
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every

parallel = args.parallel

paddle.set_device(device)
num_labels = len(label2text.keys())


random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


class LMForwardAPI:
    def __init__(self, model_name="ernie-3.0-medium-zh", n_prompt_tokens=50, loss_type="ce"):

        self.model = p_ErnieForMaskedLM.from_pretrained("ernie-3.0-medium-zh", n_prompt_tokens=n_prompt_tokens)
        self.tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-medium-zh")
        self.n_prompt_tokens = n_prompt_tokens

        self.config = self.model.ernie.init_config

        self.init_prompt = None
        self.model.eval()

        embedding = self.model.ernie.get_input_embeddings().weight.clone().cpu()

        mu_hat = np.mean(embedding.flatten().detach().cpu().numpy())
        std_hat = np.std(embedding.flatten().detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

        print("[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}".format(mu_hat, std_hat, mu, std))
        self.linear = paddle.nn.layer.Linear(
            intrinsic_dim, n_prompt_tokens * self.config["hidden_size"], bias_attr=False
        )

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.stop_tol = 15
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type

        self.metric_key = "f1"

        self.ce_loss = paddle.nn.loss.CrossEntropyLoss(reduction="mean")

    def calc_metric(self, logits, target):

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        interest_index = paddle.to_tensor(interest_index)
        logits = paddle.gather(logits, interest_index, axis=1)
        pred = logits.argmax(axis=-1)
        if self.metric_key == "acc":
            perf = (pred == converted_target).sum() / len(target)

        elif self.metric_key == "f1":
            perf = f1_score(
                converted_target.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist(), average="macro"
            )
        else:
            raise KeyError(f"[Metric] Only support [acc, f1], got {self.metric_key} instead.")

        if self.loss_type == "ce":
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == "perf":
            loss = -1 * perf
        else:
            raise KeyError(f"[Loss] Only support [ce, perf], got {self.loss_type} instead.")

        return loss, perf

    def eval(self, prompt_embedding=None):

        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt

        bsz = len(train_data["input_ids"])  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = paddle.Tensor(pe).astype(paddle.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(paddle.concat([z.reshape([1, n_prompt_tokens, -1])] * bsz))
            prompt_embedding = paddle.concat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(train_data["input_ids"])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = paddle.Tensor(prompt_embedding).astype(paddle.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az

            prompt_embedding = paddle.concat([prompt_embedding.reshape([1, n_prompt_tokens, -1])] * bsz)
        else:
            raise ValueError(
                f"[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead."
            )
        self.model.set_prompt_embedding(prompt_embedding)

        # for k, v in train_data.items():
        #     train_data[k] = v.to("cuda:1")
        with paddle.no_grad():

            logits = self.model(
                input_ids=train_data["input_ids"],
                attention_mask=train_data["attention_mask"],
                mask_pos=train_data["mask_pos"],
            )["logits"]
        loss, perf = self.calc_metric(logits, train_data["labels"])

        if perf > self.best_train_perf:
            self.best_train_perf = perf

        if self.num_call % self.print_every == 0:
            print(
                "[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}".format(
                    self.num_call, round(float(loss), 4), round(float(perf), 4), round(float(self.best_train_perf), 4)
                )
            )

        if self.num_call % self.eval_every == 0:
            print("********* Evaluated on dev set *********")

            total_logit = []
            total_label = []
            with paddle.no_grad():
                for batch in dev_data_loader:
                    dev_data = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "mask_pos": batch[2],
                        "labels": batch[3],
                    }
                    logits = self.model(
                        input_ids=dev_data["input_ids"],
                        attention_mask=dev_data["attention_mask"],
                        mask_pos=dev_data["mask_pos"],
                    )["logits"]
                    total_logit.append(logits)
                    total_label.append(dev_data["labels"])
            dev_loss, dev_perf = self.calc_metric(paddle.concat(total_logit), paddle.concat(total_label))
            # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
            if dev_perf > self.best_dev_perf:
                self.best_dev_perf = dev_perf
                self.best_prompt = copy.deepcopy(tmp_prompt)

            print(
                "Dev loss: {}. Dev perf: {}. Best dev perf: {}".format(
                    round(float(dev_loss), 4), round(float(dev_perf), 4), round(float(self.best_dev_perf), 4)
                )
            )
            print("********* Done *********")

        return loss


tokenizer = ErnieTokenizer.from_pretrained(model_name)
label_map = {
    tokenizer.encode("bad", add_special_tokens=False)["input_ids"][0]: 0,  # negative
    tokenizer.encode("great", add_special_tokens=False)["input_ids"][0]: 1,  # positive
}

# data_bundle = DataLoader(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens, label2text= label2text).my_load(['train', 'dev'])
train_ds, dev_ds = load_data(file_path + "train.tsv")

transform_fn = partial(transform_data)

train_ds = train_ds.map(transform_fn, lazy=False)
dev_ds = dev_ds.map(transform_fn, lazy=False)


def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=0),  # attention_mask
        Stack(dtype="int64"),  # masked_positions
        Stack(dtype="int64"),  # masked_lm_labels
    )
    return [data for data in fn(samples)]


trans_func = partial(
    convert_example, tokenizer=tokenizer, max_seq_length=max_seq_length, n_prompt_tokens=n_prompt_tokens
)


train_data_loader = create_dataloader(
    train_ds, mode="train", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
)
dev_data_loader = create_dataloader(
    dev_ds, mode="eval", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
)


model_forward_api = LMForwardAPI(model_name=model_name, n_prompt_tokens=n_prompt_tokens, loss_type=loss_type)

cma_opts = {
    "seed": seed,
    "popsize": popsize,
    "maxiter": budget if parallel else budget // popsize,
    "verbose": -1,
}
if bound > 0:
    cma_opts["bounds"] = [-1 * bound, 1 * bound]
es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)
print("{} Evaluation.".format("Parallel" if parallel else "Serial"))


# opt = cma.CMAOptions()
start_time = time.time()
while not es.stop():
    for step, batch in enumerate(train_data_loader, start=1):
        train_data = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "mask_pos": batch[2],
            "labels": batch[3],
        }
        solutions = es.ask()

        fitnesses = [model_forward_api.eval(x) for x in solutions]
        es.tell(solutions, fitnesses)
        if model_forward_api.best_train_perf == 1:
            model_forward_api.stop_tol -= 1
        if model_forward_api.stop_tol == 0:
            es.stop()["stop"] = 1

end_time = time.time()
print("Done. Elapsed time: {} (mins)".format((end_time - start_time) / 60))
if not os.path.exists("./results/"):
    os.makedirs("./results/")
paddle.save(
    model_forward_api.linear(paddle.to_tensor(model_forward_api.best_prompt, dtype=paddle.float32)),
    path="./results/best.pt",
)
