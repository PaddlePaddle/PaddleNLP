#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os
import random
import time
import math
import pickle
import logging

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import data
import lm_model
from args import parse_args
from utils.cards import get_cards
from utils.init import init_pretraining_params

logging.basicConfig()


def prepare_batch_input(batch, args):
    x = batch['token_ids']
    x_r = batch['token_ids_reverse']
    y = batch['next_token_id']
    y_r = batch['next_token_id_reverse']
    inst = []
    for i in range(len(x)):
        if args.use_custom_samples:
            custom_samples_array = np.zeros(
                (args.num_steps, args.n_negative_samples_batch + 1),
                dtype='int64')
            custom_samples_array_r = np.zeros(
                (args.num_steps, args.n_negative_samples_batch + 1),
                dtype='int64')
            custom_probabilities_array = np.zeros(
                (args.num_steps, args.n_negative_samples_batch + 1),
                dtype='float32')
            for j in range(args.num_steps):
                for k in range(args.n_negative_samples_batch + 1):
                    custom_samples_array[j][k] = k
                    custom_samples_array_r[j][k] = k
                    custom_probabilities_array[j][k] = 1.0
                custom_samples_array[j][0] = y[i][j]
                custom_samples_array_r[j][0] = y_r[i][j]
            inst.append([
                x[i], y[i], x_r[i], y_r[i], custom_samples_array,
                custom_samples_array_r, custom_probabilities_array
            ])
        else:
            inst.append([x[i], y[i], x_r[i], y_r[i]])
    return inst


def batch_reader(batch_list, args):
    res = []
    for batch in batch_list:
        res.append(prepare_batch_input(batch, args))
    return res


def read_multiple(reader, batch_size, count, clip_last=True):
    """
    Stack data from reader for multi-devices.
    """

    def __impl__():
        # one time read batch_size * count data for rnn
        for data in reader():
            inst_num_per_part = batch_size
            split_data = {}
            len_check = True
            for k in data.keys():
                if data[k] is not None:
                    if len(data[k]) != batch_size * count:
                        len_check = False
                        print("data check error!!, data=" + data[k] + ", k=" +
                              k)
                        break
            if len_check:
                res = []
                for i in range(count):
                    split_data = {}
                    for k in data.keys():
                        if data[k] is not None:
                            split_data[k] = data[k][inst_num_per_part * i:
                                                    inst_num_per_part * (i + 1)]
                    res.append(split_data)
                yield res

    return __impl__


def LodTensor_Array(lod_tensor):
    lod = lod_tensor.lod()
    array = np.array(lod_tensor)
    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array


def get_current_model_para(train_prog, train_exe):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    return vals


def save_para_npz(train_prog, train_exe):
    logger.info("begin to save model to model_base")
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    emb = vals["embedding_para"]
    logger.info("begin to save model to model_base")
    np.savez("mode_base", **vals)


def prepare_input(batch, epoch_id=0, with_lr=True):
    x, y = batch
    inst = []
    for i in range(len(x)):
        inst.append([x[i], y[i]])
    return inst


def eval(vocab, infer_progs, dev_count, logger, args):
    infer_prog, infer_startup_prog, infer_model = infer_progs
    feed_order = infer_model.feed_order
    loss = infer_model.loss

    # prepare device
    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    if not args.use_gpu:
        place = fluid.CPUPlace()
        import multiprocessing
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    total_loss = 0.0
    total_cnt = 0
    n_batch_cnt = 0
    n_batch_loss = 0.0
    val_feed_list = [
        infer_prog.global_block().var(var_name) for var_name in feed_order
    ]
    val_feeder = fluid.DataFeeder(val_feed_list, place)
    dev_data = data.BidirectionalLMDataset(
        args.test_path, vocab, test=True, shuffle_on_load=False)
    dev_data_iter = lambda: dev_data.iter_batches(args.batch_size * dev_count, args.num_steps)
    dev_reader = read_multiple(dev_data_iter, args.batch_size, dev_count)

    last_hidden_values = np.zeros(
        (dev_count, args.num_layers * 2 * args.batch_size * args.embed_size),
        dtype='float32')
    last_cell_values = np.zeros(
        (dev_count, args.num_layers * 2 * args.batch_size * args.hidden_size),
        dtype='float32')
    for batch_id, batch_list in enumerate(dev_reader(), 1):
        feed_data = batch_reader(batch_list, args)
        feed = list(val_feeder.feed_parallel(feed_data, dev_count))
        for i in range(dev_count):
            init_hidden_tensor = fluid.core.LoDTensor()
            if args.use_gpu:
                placex = fluid.CUDAPlace(i)
            else:
                placex = fluid.CPUPlace()
            init_hidden_tensor.set(last_hidden_values[i], placex)
            init_cell_tensor = fluid.core.LoDTensor()
            init_cell_tensor.set(last_cell_values[i], placex)

            feed[i]['init_hiddens'] = init_hidden_tensor
            feed[i]['init_cells'] = init_cell_tensor
        last_hidden_values = []
        last_cell_values = []
        for i in range(dev_count):
            val_fetch_outs = exe.run(program=infer_prog,
                                     feed=feed[i],
                                     fetch_list=[
                                         infer_model.loss.name,
                                         infer_model.last_hidden.name,
                                         infer_model.last_cell.name
                                     ],
                                     return_numpy=False)
            last_hidden_values.append(np.array(val_fetch_outs[1]))
            last_cell_values.append(np.array(val_fetch_outs[2]))
            total_loss += np.array(val_fetch_outs[0]).sum()

            n_batch_cnt += len(np.array(val_fetch_outs[0]))
            total_cnt += len(np.array(val_fetch_outs[0]))
            n_batch_loss += np.array(val_fetch_outs[0]).sum()

        last_hidden_values = np.array(last_hidden_values).reshape((
            dev_count, args.num_layers * 2 * args.batch_size * args.embed_size))
        last_cell_values = np.array(last_cell_values).reshape(
            (dev_count,
             args.num_layers * 2 * args.batch_size * args.hidden_size))

        log_every_n_batch = args.log_interval
        if log_every_n_batch > 0 and batch_id % log_every_n_batch == 0:
            logger.info('Average dev loss from batch {} to {} is {}'.format(
                batch_id - log_every_n_batch + 1, batch_id, "%.10f" % (
                    n_batch_loss / n_batch_cnt)))
            n_batch_loss = 0.0
            n_batch_cnt = 0
        batch_offset = 0

    ppl = np.exp(total_loss / total_cnt)
    return ppl


def train():
    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    args = parse_args()
    logger.info('Running with args : {}'.format(args))
    logger.info('Running paddle : {}'.format(paddle.version.commit))

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    data_path = args.data_path
    logger.info("begin to load vocab")
    vocab = data.Vocabulary(args.vocab_path, validate_file=True)
    vocab_size = vocab.size
    logger.info("finished load vocab")

    if args.enable_ce:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    logger.info('build the model...')
    # build model
    train_prog = fluid.Program()
    train_startup_prog = fluid.Program()
    if args.enable_ce:
        train_prog.random_seed = args.random_seed
        train_startup_prog.random_seed = args.random_seed
    # build infer model
    infer_prog = fluid.Program()
    infer_startup_prog = fluid.Program()
    with fluid.program_guard(infer_prog, infer_startup_prog):
        with fluid.unique_name.guard():
            # Infer process
            infer_model = lm_model.LanguageModel(
                args, vocab_size, test_mode=True)
            infer_model.build()
    infer_progs = infer_prog, infer_startup_prog, infer_model

    with fluid.program_guard(train_prog, train_startup_prog):
        with fluid.unique_name.guard():
            # Training process
            train_model = lm_model.LanguageModel(
                args, vocab_size, test_mode=False)
            train_model.build()
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=args.max_grad_norm))

            # build optimizer
            if args.optim == 'adagrad':
                optimizer = fluid.optimizer.Adagrad(
                    learning_rate=args.learning_rate,
                    epsilon=0.0,
                    initial_accumulator_value=1.0)
            elif args.optim == 'sgd':
                optimizer = fluid.optimizer.SGD(
                    learning_rate=args.learning_rate)
            elif args.optim == 'adam':
                optimizer = fluid.optimizer.Adam(
                    learning_rate=args.learning_rate)
            elif args.optim == 'rprop':
                optimizer = fluid.optimizer.RMSPropOptimizer(
                    learning_rate=args.learning_rate)
            else:
                logger.error('Unsupported optimizer: {}'.format(args.optim))
                exit(-1)
            optimizer.minimize(train_model.loss * args.num_steps)
            # initialize parameters
            place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
            exe = Executor(place)
    train_progs = train_prog, train_startup_prog, train_model

    if args.local:
        logger.info("local start_up:")
        train_loop(args, logger, vocab, train_progs, infer_progs, optimizer)
    else:
        if args.update_method == "nccl2":
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            if args.test_nccl:
                worker_endpoints_env = os.getenv("PADDLE_WORK_ENDPOINTS")
                worker_endpoints = worker_endpoints_env.split(',')
                trainers_num = len(worker_endpoints)
                current_endpoint = worker_endpoints[trainer_id]
            else:
                port = os.getenv("PADDLE_PORT")
                worker_ips = os.getenv("PADDLE_TRAINERS")
                worker_endpoints = []
                for ip in worker_ips.split(","):
                    worker_endpoints.append(':'.join([ip, port]))
                worker_endpoints_env = ','.join(worker_endpoints)
                trainers_num = len(worker_endpoints)
                current_endpoint = os.getenv("POD_IP") + ":" + port
            if trainer_id == 0:
                logger.info("train_id == 0, sleep 60s")
                time.sleep(60)

            logger.info("trainers_num:{}".format(trainers_num))
            logger.info("worker_endpoints:{}".format(worker_endpoints))
            logger.info("current_endpoint:{}".format(current_endpoint))
            config = fluid.DistributeTranspilerConfig()
            config.mode = "nccl2"
            t = fluid.DistributeTranspiler(config=config)
            t.transpile(
                trainer_id,
                trainers=worker_endpoints_env,
                current_endpoint=current_endpoint,
                program=train_prog,
                startup_program=train_startup_prog)
            train_progs = train_prog, train_startup_prog, train_model
            train_loop(args, logger, vocab, train_progs, infer_progs, optimizer,
                       trainers_num, trainer_id, worker_endpoints)
        else:
            port = os.getenv("PADDLE_PORT", "6174")
            pserver_ips = os.getenv("PADDLE_PSERVERS")
            eplist = []
            for ip in pserver_ips.split(","):
                eplist.append(':'.join([ip, port]))
            pserver_endpoints = ",".join(eplist)
            trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
            current_endpoint = os.getenv("POD_IP") + ":" + port
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))

            logger.info("pserver_endpoints:{}".format(pserver_endpoints))
            logger.info("current_endpoint:{}".format(current_endpoint))
            logger.info("trainer_id:{}".format(trainer_id))
            logger.info("pserver_ips:{}".format(pserver_ips))
            logger.info("port:{}".format(port))

            t = fluid.DistributeTranspiler()
            t.transpile(
                trainer_id,
                pservers=pserver_endpoints,
                trainers=trainers,
                program=train_prog,
                startup_program=startup_prog)

            if training_role == "PSERVER":
                logger.info("distributed: pserver started")
                current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                    "PADDLE_PORT")
                if not current_endpoint:
                    logger.critical("need env SERVER_ENDPOINT")
                    exit(1)
                pserver_prog = t.get_pserver_program(current_endpoint)
                pserver_startup = t.get_startup_program(current_endpoint,
                                                        pserver_prog)

                exe.run(pserver_startup)
                exe.run(pserver_prog)
            elif training_role == "TRAINER":
                logger.info("distributed: trainer started")
                trainer_prog = t.get_trainer_program()
                train_loop(args, logger, vocab, train_progs, infer_progs,
                           optimizer)
            else:
                logger.critical(
                    "environment var TRAINER_ROLE should be TRAINER os PSERVER")
                exit(1)


def train_loop(args,
               logger,
               vocab,
               train_progs,
               infer_progs,
               optimizer,
               nccl2_num_trainers=1,
               nccl2_trainer_id=0,
               worker_endpoints=None):
    train_prog, train_startup_prog, train_model = train_progs
    infer_prog, infer_startup_prog, infer_model = infer_progs

    # prepare device
    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    if not args.use_gpu:
        place = fluid.CPUPlace()
        import multiprocessing
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    if args.load_dir:
        logger.info('load pretrained checkpoints from {}'.format(args.load_dir))
        fluid.io.load_persistables(exe, args.load_dir, main_program=train_prog)
    elif args.load_pretraining_params:
        logger.info('load pretrained params from {}'.format(
            args.load_pretraining_params))
        exe.run(train_startup_prog)
        init_pretraining_params(
            exe, args.load_pretraining_params, main_program=train_prog)
    else:
        exe.run(train_startup_prog)

    # prepare data
    feed_list = [
        train_prog.global_block().var(var_name)
        for var_name in train_model.feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    logger.info('Training the model...')
    exe_strategy = fluid.parallel_executor.ExecutionStrategy()
    parallel_executor = fluid.ParallelExecutor(
        loss_name=train_model.loss.name,
        main_program=train_prog,
        use_cuda=bool(args.use_gpu),
        exec_strategy=exe_strategy,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)

    logger.info("begin to load data")
    train_data = data.BidirectionalLMDataset(
        args.train_path,
        vocab,
        test=(not args.shuffle),
        shuffle_on_load=args.shuffle)
    logger.info("finished load vocab")

    # get train epoch size
    log_interval = args.log_interval
    total_time = 0.0
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    custom_samples_array = np.zeros(
        (batch_size, args.num_steps, args.n_negative_samples_batch + 1),
        dtype='int64')
    custom_probabilities_array = np.zeros(
        (batch_size, args.num_steps, args.n_negative_samples_batch + 1),
        dtype='float32')
    for i in range(batch_size):
        for j in range(0, args.num_steps):
            for k in range(0, args.n_negative_samples_batch + 1):
                custom_samples_array[i][j][k] = k
                custom_probabilities_array[i][j][k] = 1.0

    start_time = time.time()
    train_data_iter = lambda: train_data.iter_batches(batch_size * dev_count, args.num_steps)
    train_reader = read_multiple(train_data_iter, batch_size, dev_count)
    total_num = 0
    n_batch_loss = 0.0
    n_batch_cnt = 0
    last_hidden_values = np.zeros(
        (dev_count, args.num_layers * 2 * batch_size * args.embed_size),
        dtype='float32')
    last_cell_values = np.zeros(
        (dev_count, args.num_layers * 2 * batch_size * hidden_size),
        dtype='float32')
    n_tokens_per_batch = args.batch_size * args.num_steps
    n_batches_per_epoch = int(args.all_train_tokens / n_tokens_per_batch)
    n_batches_total = args.max_epoch * n_batches_per_epoch
    begin_time = time.time()
    ce_info = []
    final_batch_id = 0
    for batch_id, batch_list in enumerate(train_reader(), 1):
        if batch_id > n_batches_total:
            break
        final_batch_id = batch_id
        feed_data = batch_reader(batch_list, args)
        feed = list(feeder.feed_parallel(feed_data, dev_count))
        for i in range(dev_count):
            init_hidden_tensor = fluid.core.LoDTensor()
            if args.use_gpu:
                placex = fluid.CUDAPlace(i)
            else:
                placex = fluid.CPUPlace()
            init_hidden_tensor.set(last_hidden_values[i], placex)
            init_cell_tensor = fluid.core.LoDTensor()
            init_cell_tensor.set(last_cell_values[i], placex)

            feed[i]['init_hiddens'] = init_hidden_tensor
            feed[i]['init_cells'] = init_cell_tensor

        fetch_outs = parallel_executor.run(feed=feed,
                                           fetch_list=[
                                               train_model.loss.name,
                                               train_model.last_hidden.name,
                                               train_model.last_cell.name
                                           ],
                                           return_numpy=False)
        cost_train = np.array(fetch_outs[0]).mean()
        last_hidden_values = np.array(fetch_outs[1])
        last_hidden_values = last_hidden_values.reshape(
            (dev_count, args.num_layers * 2 * batch_size * args.embed_size))
        last_cell_values = np.array(fetch_outs[2])
        last_cell_values = last_cell_values.reshape(
            (dev_count, args.num_layers * 2 * batch_size * args.hidden_size))

        total_num += args.batch_size * dev_count
        n_batch_loss += np.array(fetch_outs[0]).sum()
        n_batch_cnt += len(np.array(fetch_outs[0]))

        if batch_id > 0 and batch_id % log_interval == 0:
            smoothed_ppl = np.exp(n_batch_loss / n_batch_cnt)
            ppl = np.exp(
                np.array(fetch_outs[0]).sum() / len(np.array(fetch_outs[0])))
            used_time = time.time() - begin_time
            speed = log_interval / used_time
            logger.info(
                "[train] step:{}, loss:{:.3f}, ppl:{:.3f}, smoothed_ppl:{:.3f}, speed:{:.3f}".
                format(batch_id, n_batch_loss / n_batch_cnt, ppl, smoothed_ppl,
                       speed))
            ce_info.append([n_batch_loss / n_batch_cnt, used_time])
            n_batch_loss = 0.0
            n_batch_cnt = 0
            begin_time = time.time()
        if batch_id > 0 and batch_id % args.dev_interval == 0:
            valid_ppl = eval(vocab, infer_progs, dev_count, logger, args)
            logger.info("valid ppl {}".format(valid_ppl))
        if batch_id > 0 and batch_id % args.save_interval == 0:
            epoch_id = int(batch_id / n_batches_per_epoch)
            model_path = os.path.join(args.para_save_dir,
                                      str(batch_id + epoch_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(
                executor=exe, dirname=model_path, main_program=train_prog)

    if args.enable_ce:
        card_num = get_cards()
        ce_loss = 0
        ce_time = 0
        try:
            ce_loss = ce_info[-2][0]
            ce_time = ce_info[-2][1]
        except:
            print("ce info error")
        print("kpis\ttrain_duration_card%s\t%s" % (card_num, ce_time))
        print("kpis\ttrain_loss_card%s\t%f" % (card_num, ce_loss))

    end_time = time.time()
    total_time += end_time - start_time
    epoch_id = int(final_batch_id / n_batches_per_epoch)
    model_path = os.path.join(args.para_save_dir, str(epoch_id))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    fluid.io.save_persistables(
        executor=exe, dirname=model_path, main_program=train_prog)
    valid_ppl = eval(vocab, infer_progs, dev_count, logger, args)
    logger.info("valid ppl {}".format(valid_ppl))
    test_ppl = eval(vocab, infer_progs, dev_count, logger, args)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    train()
