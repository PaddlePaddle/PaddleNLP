# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddlenlp.trl.llm_utils import get_rotary_position_embedding
from paddlenlp_ops import step_paddle
from server.data.processor import DataProcessor
from server.engine.config import Config
from server.utils import get_logger
from task_queue_manager import TaskQueueManager

File_Path = os.path.realpath(sys.argv[0])
Dir_Path = os.path.dirname(File_Path)
logger = get_logger("infer_server", "infer.log")


class ModelRunner:
    def __init__(self, args):
        self.args = args

        # 2**63 - 1
        self.MAX_INFER_SEED = 9223372036854775806

        self.config = Config()
        self.model_cfg = self.config.get_model_config()
        self.format_print_configuration()

        self.args.num_layers = self.get_value(self.model_cfg, ["num_hidden_layers", "num_layers"])
        self.args.num_attention_heads = self.get_value(self.model_cfg, ["num_attention_heads", "n_head"])
        self.args.hidden_size = self.model_cfg["hidden_size"]

        self.nranks = dist.get_world_size()
        self.init_dist_env()
        self.rank = fleet.worker_index()

        self.load_model_init_val()

        self.share_inputs = {}
        self.cache_kvs = {}
        self.init_inputs()

        self.infer_queue = TaskQueueManager(rank=self.rank, mp_num=self.nranks, port=self.config.infer_port)

        model_rank_path = os.path.join(self.args.model_dir, f"rank_{self.rank}")
        if not os.path.exists(model_rank_path):
            model_rank_path = self.args.model_dir

        self.infer_engine = InferenceEngine(model_dir=model_rank_path,
                                            share_inputs=self.share_inputs,
                                            cache_kvs=self.cache_kvs,
                                            config=self.config,
                                            mp_degree=self.nranks
                                        )

    def read_model_config(self):
        """
        load model config file from json file

        Returns:
            model_config_json: dict, model config file
        """
        model_config_json = json.load(open(self.config_file, 'r', encoding='utf-8'))
        return model_config_json

    def get_value(self, cfg, names):
        """
        get value from config file by key names
        """
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if name in cfg:
                return cfg[name]
            break
        raise Exception(
            "Cannot find any one of key in {} in configuration file.".format(
                names))

    def format_print_configuration(self):
        """
        print model config
        """
        logger.info("===============   Model Information   ==============")
        for k, v in self.model_cfg.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============== Service Configuration ===============")
        for k, v in vars(self.args).items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=====================================================\n")

    def load_model_init_val(self):
        """
        initialize model config from config file
        """
        self.top_p = self.model_cfg.get("top_p", 0.0)
        self.temperature = self.model_cfg.get("temperature", 1.0)
        self.rope_theta = self.model_cfg.get('rope_theta', 10000.0)
        self.rope_scaling = self.model_cfg.get('rope_scaling', None)
        self.penalty_score = self.model_cfg.get('penalty_score', 1.0)
        self.frequency_score = self.model_cfg.get('frequency_score', 0.0)
        self.presence_score = self.model_cfg.get('presence_score', 0.0)
        self.min_length = self.model_cfg.get('min_length', 1)
        self.max_length = self.model_cfg.get('max_length', 1024)

        data_processor = DataProcessor()
        # reserve an eos token for request
        self.eos_tokens_lens = data_processor.get_eos_tokens_lens() + 1
        self.pad_token_id = data_processor.get_pad_id()

    def init_dist_env(self, seed=20):
        """
        init distributed env
        """
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.nranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)

    def init_inputs(self):
        # init all inputs
        if "num_key_value_heads" in self.model_cfg and \
                self.model_cfg["num_key_value_heads"] is not None and \
                int(self.model_cfg["num_key_value_heads"]) > 0:
            kv_num_head = int(self.model_cfg["num_key_value_heads"]) // self.nranks
        else:
            kv_num_head = self.args.num_attention_heads // self.nranks

        for i in range(self.args.num_layers):
            if not self.args.use_cache_kv_int8:
                cache_type = self.args.dtype
            else:
                cache_type = "uint8"

            self.cache_kvs["key_caches_{}".format(i)] = paddle.full(shape=[
                self.args.max_block_num, kv_num_head,
                self.args.block_size, self.args.hidden_size // self.args.num_attention_heads
            ], fill_value=0, dtype=cache_type)
            self.cache_kvs["value_caches_{}".format(i)] = paddle.full(shape=[
                self.args.max_block_num, kv_num_head,
                self.args.block_size, self.args.hidden_size // self.args.num_attention_heads
            ], fill_value=0, dtype=cache_type)

        pre_max_block_num = (self.args.max_seq_len + self.args.block_size - 1) // self.args.block_size + self.args.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full(
                        shape=[self.args.max_batch_size, pre_max_block_num], fill_value=-1, dtype="int32")

        self.share_inputs['pre_ids'] = paddle.to_tensor(
                        np.full((self.args.max_batch_size, self.args.max_dec_len), -1, dtype='int64'))

        tmp_position_ids = paddle.arange(self.args.max_seq_len).reshape((1, -1))
        self.share_inputs['rope_emb'] = get_rotary_position_embedding(tmp_position_ids,
                        self.args.hidden_size // self.args.num_attention_heads,
                        self.rope_theta, self.rope_scaling)
        self.share_inputs['input_ids'] = paddle.full(
                        shape=[self.args.max_batch_size, self.args.max_seq_len],
                        fill_value=self.pad_token_id, dtype='int64')
        self.share_inputs['top_p'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.top_p, dtype="float32")
        self.share_inputs['temperature'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.temperature, dtype="float32")
        self.share_inputs['eos_token_id'] = paddle.to_tensor(
                            np.zeros((self.eos_tokens_lens, 1)).reshape(-1, 1).astype("int64"))
        self.share_inputs['penalty_score'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.penalty_score, dtype="float32")
        self.share_inputs['frequency_score'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.frequency_score, dtype="float32")
        self.share_inputs['presence_score'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.presence_score, dtype="float32")
        self.share_inputs['seq_lens_this_time'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int32")
        self.share_inputs['seq_lens_encoder'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int32")
        self.share_inputs['step_seq_lens_encoder'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int32")
        self.share_inputs['seq_lens_decoder'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int32")
        self.share_inputs['step_idx'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int64")
        self.share_inputs['min_length'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.min_length, dtype="int64")
        self.share_inputs['max_length'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=self.max_length, dtype="int64")
        self.share_inputs['not_need_stop'] = paddle.full(
                            shape=[1], fill_value=False, dtype="bool")
        self.share_inputs['stop_flags'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=True, dtype="bool")
        self.share_inputs['stop_nums'] = paddle.full(
                            shape=[1], fill_value=self.args.max_batch_size, dtype="int64")
        self.share_inputs['bad_tokens'] = paddle.full(
                            shape=[1], fill_value=-1, dtype="int64")
        self.share_inputs['next_tokens'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=-1, dtype="int64")
        self.share_inputs['is_block_step'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=False, dtype="bool")
        self.share_inputs['encoder_block_lens'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=0, dtype="int32")
        self.share_inputs['step_block_list'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=-1, dtype="int32")
        self.share_inputs['step_lens'] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.share_inputs['recover_block_list'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=-1, dtype="int32")
        self.share_inputs['recover_lens'] = paddle.full(
                            shape=[1], fill_value=0, dtype="int32")
        self.share_inputs['need_block_list'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=-1, dtype="int32")
        self.share_inputs['need_block_len'] = paddle.full(
                            shape=[1], fill_value=0, dtype="int32")
        self.share_inputs['used_list_len'] = paddle.full(
                            shape=[self.args.max_batch_size], fill_value=0, dtype="int32")
        self.share_inputs['infer_seed'] = paddle.full(
                            shape=[self.args.max_batch_size, 1], fill_value=0, dtype="int64")
        free_list = list(range(int(self.args.max_block_num * self.args.block_ratio)))
        self.free_list_len = len(free_list)
        self.share_inputs['free_list'] = paddle.to_tensor(free_list, dtype="int32")
        self.share_inputs['free_list_len'] = paddle.full(
                            shape=[1], fill_value=self.free_list_len, dtype="int32")

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """
        for i in range(len(tasks)):
            task = tasks[i]
            idx = task['idx']
            length = len(task['input_ids'])
            self.share_inputs['input_ids'][idx:idx + 1, :length] = np.array(task['input_ids'])
            if len(task['eos_token_ids']) < self.eos_tokens_lens:
                task['eos_token_ids'].append(task['eos_token_ids'][0])
            self.share_inputs['eos_token_id'][:] = np.array(task['eos_token_ids'], dtype="int64").reshape(-1, 1)
            self.share_inputs['pre_ids'][idx:idx + 1] = -1
            self.share_inputs['top_p'][idx:idx + 1] = task.get('topp', 0.7)
            self.share_inputs['temperature'][idx:idx + 1] = task.get('temperature', 0.95)
            self.share_inputs['penalty_score'][idx:idx + 1] = task.get('penalty_score', 1.0)
            self.share_inputs['frequency_score'][idx:idx + 1] = task.get('frequency_score', 0.0)
            self.share_inputs['presence_score'][idx:idx + 1] = task.get('presence_score', 0.0)
            self.share_inputs['seq_lens_this_time'][idx:idx + 1] = length
            self.share_inputs['step_seq_lens_encoder'][idx:idx + 1] = length
            self.share_inputs['seq_lens_encoder'][idx:idx + 1] = length
            self.share_inputs['seq_lens_decoder'][idx:idx + 1] = 0
            self.share_inputs['step_idx'][idx:idx + 1] = 0
            self.share_inputs['min_length'][idx:idx + 1] = task.get('min_dec_len', 1)
            if "max_dec_len" in task:
                max_dec_len = task['max_dec_len']
            elif "seq_len" in task:
                max_dec_len = task['seq_len']
            else:
                max_dec_len = self.args.max_dec_len
            self.share_inputs['max_length'][idx:idx + 1] = max_dec_len
            self.share_inputs['stop_flags'][idx:idx + 1] = False

            if "infer_seed" in task:
                self.share_inputs['infer_seed'][idx:idx + 1] = task['infer_seed']

            encoder_block_num = len(task['block_tables'])
            self.share_inputs['encoder_block_lens'][idx:idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][idx:idx + 1, :encoder_block_num] = np.array(
                                            task['block_tables'], dtype="int32")

    def step_cuda(self, seq_lens_this_time):
        """
        step cuda
        """
        step_paddle(self.share_inputs['stop_flags'], seq_lens_this_time,
                    self.share_inputs['step_seq_lens_encoder'],
                    self.share_inputs['seq_lens_encoder'],
                    self.share_inputs['seq_lens_decoder'], self.share_inputs["block_tables"],
                    self.share_inputs['encoder_block_lens'],
                    self.share_inputs["is_block_step"], self.share_inputs['step_block_list'],
                    self.share_inputs['step_lens'], self.share_inputs['recover_block_list'],
                    self.share_inputs['recover_lens'], self.share_inputs['need_block_list'],
                    self.share_inputs['need_block_len'], self.share_inputs['used_list_len'],
                    self.share_inputs['free_list'], self.share_inputs['free_list_len'],
                    self.share_inputs['input_ids'], self.share_inputs['pre_ids'],
                    self.share_inputs['step_idx'], self.share_inputs['next_tokens'],
                    self.args.block_size, self.args.enc_dec_block_num, self.args.first_token_id)

    def initialize_engine_ready_check_flag(self):
        """
        initialize engine ready flag in shared memory

        Returns:
            shm_engine_ready_check_flag: engine ready flag
            engine_ready_check_flag_array: engine ready flag array
        """
        engine_ready_check_flag = np.zeros([1], dtype=np.int32)
        shm_engine_ready_check_flag = shared_memory.SharedMemory(
                        name=self.config.get_unique_name("engine_ready_check_flag"))
        engine_ready_check_flag_array = np.ndarray(engine_ready_check_flag.shape,
                                            dtype=engine_ready_check_flag.dtype,
                                            buffer=shm_engine_ready_check_flag.buf)
        return shm_engine_ready_check_flag, engine_ready_check_flag_array

    def initialize_engine_live_flag(self):
        """
        initialize infer live flag in shared memory

        Returns:
            infer_live_flag_shm: infer live flag
        """
        infer_live_flag_shm = shared_memory.SharedMemory(create=True,
                                            size=1,
                                            name=self.config.get_unique_name("shm_flag_infer_{}_live".format(self.rank)))
        return infer_live_flag_shm

    def initialize_engine_healthy_recorded_time_flag(self):
        """
        initialize engine healthy recorded time flag in shared memory

        Returns:
            shm_engine_healthy_recorded_time: engine healthy recorded time flag
        """
        engine_healthy_recorded_time = np.zeros([1], dtype=float)
        shm_engine_healthy_recorded_time = shared_memory.SharedMemory(
                    name=self.config.get_unique_name("engine_healthy_recorded_time"))
        engine_healthy_recorded_time_array = np.ndarray(engine_healthy_recorded_time.shape,
                                            dtype=engine_healthy_recorded_time.dtype,
                                            buffer=shm_engine_healthy_recorded_time.buf)
        return shm_engine_healthy_recorded_time, engine_healthy_recorded_time_array

    def run(self):
        """
        run infer
        """
        flag_array = np.zeros([1], dtype=np.int32)
        shm_flag_broadcast = shared_memory.SharedMemory(
                        name=self.config.get_unique_name("shm_pd_infer_flag_broadcast"))
        flag_broadcast_array = np.ndarray(flag_array.shape,
                                            dtype=flag_array.dtype,
                                            buffer=shm_flag_broadcast.buf)

        flag_array = np.zeros([self.nranks], dtype=np.int32)
        shm_flag_ready = shared_memory.SharedMemory(name=self.config.get_unique_name("shm_flag_infer_ready"))
        flag_ready_array = np.ndarray(flag_array.shape,
                                        dtype=flag_array.dtype,
                                        buffer=shm_flag_ready.buf)
        flag_ready_array[self.rank] = 1

        flag_array = np.zeros([1], dtype=np.int32)
        shm_flag_has_block_step = shared_memory.SharedMemory(name=self.config.get_unique_name("shm_flag_has_block_step"))
        flag_has_block_step_array = np.ndarray(flag_array.shape,
                                                dtype=flag_array.dtype,
                                                buffer=shm_flag_has_block_step.buf)

        use_custom_health_checker = self.config.use_custom_health_checker
        if use_custom_health_checker:
            shm_engine_ready_check_flag_array, engine_ready_check_flag_array = self.initialize_engine_ready_check_flag()
            engine_ready_check_flag_array[0] = 1
            shm_engine_healthy_recorded_time_array, engine_healthy_recorded_time_array = self.initialize_engine_healthy_recorded_time_flag()
            engine_healthy_recorded_time_array[0] = time.time()
            infer_live_flag_shm = self.initialize_engine_live_flag()
        infer_seed_increment = paddle.full(shape=[self.args.max_batch_size, 1],
                                            fill_value=4,
                                            dtype="int64")
        thread_executor = ThreadPoolExecutor(max_workers=1)
        seq_lens_this_time = None
        real_bsz = None

        while True:
            if use_custom_health_checker:
                engine_healthy_recorded_time_array[0] = time.time()

            if self.rank == 0:
                if not self.infer_queue.empty():
                    flag_broadcast_array[0] = 1

            if self.nranks > 1:
                paddle.distributed.barrier()

            if flag_broadcast_array[0] == 1:
                logger.info(f'rank: {self.rank} start to get')
                if seq_lens_this_time is not None:
                    self.share_inputs["seq_lens_this_time"][:real_bsz] = seq_lens_this_time

                tasks, read_finish = self.infer_queue.get()
                if read_finish:
                    flag_broadcast_array[0] = 0

                req_dicts = []
                for req_dict, bsz in tasks:
                    real_bsz = int(bsz)
                    req_dicts.extend(req_dict)
                    logger.info(
                        f'rank: {self.rank}, real_bsz: {real_bsz}, query_num: {len(req_dicts)}'
                    )

                self.dy_input_preprocess(req_dicts)
                seq_lens_this_time = copy.deepcopy(
                    self.share_inputs['seq_lens_this_time'][:real_bsz])
                self.infer_engine.seq_lens_handle.share_external_data(
                    seq_lens_this_time)
                self.share_inputs['not_need_stop'][0] = True

            if not self.share_inputs['not_need_stop']:
                if self.nranks > 1:
                    paddle.distributed.barrier()

                time.sleep(0.001)
                continue

            self.infer_engine.predictor.run()
            self.share_inputs['infer_seed'].add_(infer_seed_increment)
            self.share_inputs['infer_seed'][:] %= self.MAX_INFER_SEED
            if self.free_list_len > 0:
                self.step_cuda(seq_lens_this_time)


class InferenceEngine(object):
    """
    Model Parallel Inference Engine

    Args:
        model_dir (string): root directory of inference model
        mp_degree (int): model parallel size
    """
    def __init__(self, model_dir, share_inputs, cache_kvs, config, mp_degree=1):
        self.config = config
        self.model_dir = model_dir
        self.mp_degree = mp_degree

        self.share_inputs = share_inputs
        self.cache_kvs = cache_kvs

        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()

        self._init_predictor()
        self.share_data()

    def _init_predictor(self):
        """
        predictor init
        """
        device_id = self.rank % 8
        self.model_file = os.path.join(self.model_dir, f"model.pdmodel")
        self.param_file = os.path.join(self.model_dir, f"model.pdiparams")
        config = paddle.inference.Config(self.model_file, self.param_file)

        config.switch_ir_optim(False)
        config.enable_use_gpu(100, device_id)

        # distributed config
        if self.mp_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]
            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)
            if self.config.distributed_config_path:
                 dist_config.set_comm_init_config(self.config.distributed_config_path)
            else:
                raise Exception("Please set DISTRIBUTED_CONFIG env variable.")
                logger.warning(
                    f"Use default distributed config, please set env DISTRIBUTED_CONFIG"
                )
                dist_config.set_comm_init_config(
                    os.path.join(Dir_Path + "/config", "rank_mapping_mp{}.csv".format(self.nranks)))

            config.set_dist_config(dist_config)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_names = self.predictor.get_input_names()
        self.seq_lens_handle = self.predictor.get_input_handle('seq_lens_this_time')

    def share_data(self):
        """
        share data
        """
        for name in self.input_names:
            if "caches" in name:
                input_tensor = self.predictor.get_input_handle(name)
                input_tensor.share_external_data(self.cache_kvs[name])
                continue
            if "seq_lens_this_time" in name:
                continue
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.share_external_data(self.share_inputs[name])

    def predict(self, real_bsz):
        """
        predict
        """
        seq_lens_this_time = copy.deepcopy(
            self.share_inputs['seq_lens_this_time'][:real_bsz])
        self.seq_lens_handle.share_external_data(seq_lens_this_time)
        self.share_inputs['not_need_stop'][0] = True
        while self.share_inputs['not_need_stop']:
            self.predictor.run()
        self.share_inputs["seq_lens_this_time"][:real_bsz] = seq_lens_this_time


def parse_args():
    """
    parse args from command line
    """
    parser = argparse.ArgumentParser("Deploy LLM Inference")
    parser.add_argument('-m',
                        '--model_dir',
                        type=str,
                        default='./output',
                        help='model dir')
    parser.add_argument('-mp',
                        '--mp_degree',
                        type=int,
                        default=1,
                        help='mp degree')
    parser.add_argument('-mbs',
                        '--max_batch_size',
                        type=int,
                        default=34,
                        help='max batch size')
    parser.add_argument('--max_block_num', type=int, default=2000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=3072,
                        help='max_seq_len')
    parser.add_argument('--max_dec_len',
                        type=int,
                        default=1024,
                        help='max_dec_len')
    parser.add_argument('--use_cache_kv_int8',
                        type=int,
                        default=0,
                        help='use cache kv int8')
    parser.add_argument('--dtype',
                        type=str,
                        default="bfloat16",
                        help='input dtype')
    parser.add_argument('--enc_dec_block_num',
                        type=int,
                        default=1,
                        help="encoder's decoder num")
    parser.add_argument('--block_ratio',
                        type=float,
                        default=0.7,
                        help="block ratio")
    parser.add_argument('--first_token_id',
                        type=int,
                        default=1,
                        help="first token id")
    args = parser.parse_args()
    return args


def main():
    """
    start model runner
    """
    args = parse_args()
    model_runner = ModelRunner(args)
    model_runner.run()


if __name__ == "__main__":
    main()
