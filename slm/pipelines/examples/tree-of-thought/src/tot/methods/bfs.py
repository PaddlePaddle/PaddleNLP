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

import itertools
import logging
from functools import partial

import numpy as np
from src.tot.models import gpt


def get_value(task, x, y, n_evaluate_sample, cache_value=True, chatter=None, args=None):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None, chatter=chatter, args=chatter)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True, chatter=None, args=None):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value, chatter=chatter, args=args)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_evaluate_sample, chatter=None, args=None):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None, chatter=chatter, args=args)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y, chatter=None, args=None):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None, args=args, chatter=chatter)[0].split("\n")
    return [y + _ + "\n" for _ in proposals]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, chatter=None, args=None):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    samples = gpt(prompt, n=n_generate_sample, stop=stop, chatter=chatter, args=args)
    return [y + _ for _ in samples]


def solve(args, task, idx, to_print=True, chatter=None):
    global gpt
    if chatter:
        chatter.query = []

    gpt = partial(gpt, model=args.backend, temperature=args.temperature, args=args, chatter=chatter)
    logging.info(gpt)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == "sample":
            new_ys = [
                get_samples(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                    chatter=chatter,
                    args=args,
                )
                for y in ys
            ]
        elif args.method_generate == "propose":
            new_ys = [get_proposals(task, x, y, chatter=chatter, args=args) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == "vote":
            values = get_votes(task, x, new_ys, args.n_evaluate_sample, chatter=chatter)
        elif args.method_evaluate == "value":
            values = get_values(task, x, new_ys, args.n_evaluate_sample, chatter=chatter)

        # selection
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))

        infos.append(
            {
                "step": step,
                "x": x,
                "ys": ys,
                "new_ys": new_ys,
                "values": values,
                "select_new_ys": select_new_ys,
            }
        )
        ys = select_new_ys

    if args.query_fp and chatter:
        f = open(args.query_fp, "w", encoding="utf8")
        f.write(str(chatter.query))
        f.close()

    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True, chatter=None):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature, args=args, chatter=chatter)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, "", args.n_generate_sample, args.prompt_sample, stop=None, chatter=chatter, args=args)
    return ys, {}
