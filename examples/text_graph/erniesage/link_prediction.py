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

import os
import io
import random
import time
import argparse
from functools import partial

import numpy as np
import yaml
import paddle
import pgl
from easydict import EasyDict as edict
from paddlenlp.utils.log import logger

from models import ErnieSageForLinkPrediction
from data import TrainData, PredictData, GraphDataLoader, batch_fn


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    paddle.seed(config.seed)


def load_data(graph_data_path):
    base_graph = pgl.Graph.load(graph_data_path)
    term_ids = np.load(
        os.path.join(graph_data_path, "term_ids.npy"), mmap_mode="r")
    return base_graph, term_ids


def do_train(config):
    paddle.set_device("gpu" if config.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    base_graph, term_ids = load_data(config.graph_work_path)
    collate_fn = partial(
        batch_fn,
        samples=config.samples,
        base_graph=base_graph,
        term_ids=term_ids)

    mode = 'train'
    train_ds = TrainData(config.graph_work_path)
    model = ErnieSageForLinkPrediction.from_pretrained(
        config.model_name_or_path, config=config)
    model = paddle.DataParallel(model)

    train_loader = GraphDataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.sample_workers,
        collate_fn=collate_fn)

    optimizer = paddle.optimizer.Adam(
        learning_rate=config.lr, parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    for epoch in range(config.epoch):
        for step, (graphs, datas) in enumerate(train_loader):
            global_step += 1
            loss, outputs = model(graphs, datas)
            if global_step % config.log_per_step == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       config.log_per_step / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            if global_step % config.save_per_step == 0:
                if (not config.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(config.output_path,
                                              "model_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model._layers.save_pretrained(output_dir)
    if (not config.n_gpu > 1) or paddle.distributed.get_rank() == 0:
        output_dir = os.path.join(config.output_path, "last")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model._layers.save_pretrained(output_dir)


def tostr(data_array):
    return " ".join(["%.5lf" % d for d in data_array])


@paddle.no_grad()
def do_predict(config):
    paddle.set_device("gpu" if config.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    mode = 'predict'
    num_nodes = int(
        np.load(os.path.join(config.graph_work_path, "num_nodes.npy")))

    base_graph, term_ids = load_data(config.graph_work_path)
    collate_fn = partial(
        batch_fn,
        samples=config.samples,
        base_graph=base_graph,
        term_ids=term_ids)

    model = ErnieSageForLinkPrediction.from_pretrained(
        config.infer_model, config=config)

    model = paddle.DataParallel(model)
    predict_ds = PredictData(num_nodes)

    predict_loader = GraphDataLoader(
        predict_ds,
        batch_size=config.infer_batch_size,
        shuffle=True,
        num_workers=config.sample_workers,
        collate_fn=collate_fn)

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    id2str = io.open(
        os.path.join(config.graph_work_path, "terms.txt"),
        encoding=config.encoding).readlines()
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    fout = io.open(
        "%s/part-%s" % (config.output_path, trainer_id), "w", encoding="utf8")

    global_step = 0
    epoch = 0
    tic_train = time.time()
    model.eval()
    for step, (graphs, datas) in enumerate(predict_loader):
        global_step += 1
        loss, outputs = model(graphs, datas)
        for user_feat, user_real_index in zip(outputs[0].numpy(),
                                              outputs[3].numpy()):
            sri = id2str[int(user_real_index)].strip("\n")
            line = "{}\t{}\n".format(sri, tostr(user_feat))
            fout.write(line)
        if global_step % config.log_per_step == 0:
            logger.info(
                "predict step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (global_step, epoch, step, loss,
                   config.log_per_step / (time.time() - tic_train)))
            tic_train = time.time()
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    logger.info(config)
    if args.do_predict:
        do_func = do_predict
    else:
        do_func = do_train

    if config.n_gpu > 1 and paddle.fluid.core.is_compiled_with_cuda(
    ) and paddle.fluid.core.get_cuda_device_count() > 1:
        paddle.distributed.spawn(do_func, args=(config, ), nprocs=config.n_gpu)
    else:
        do_func(config)
