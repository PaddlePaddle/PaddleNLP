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

import json
import os
import sys
from datetime import datetime

from paddlenlp.generation import GenerationConfig
from server.utils import model_server_logger
from dataclasses import dataclass


class Config:
    """
    initial configuration
    """

    def __init__(self):
        self.read_from_env()

    def read_from_env(self):
        """
        get the configuration from environment
        """
        env = os.environ
        self.model_dir = env.get(
            "MODEL_DIR", "/opt/output/Serving/models")
        if not self.model_dir:
            raise Exception("The parameter MODEL_DIR is None.")
        self.mp_num = int(env.get("MP_NUM", 8))
        self.config_json_file = env.get("CONFIG_JSON_FILE", "config.json")
        self.model_config_path = os.path.join(self.model_dir, self.config_json_file)
        if env.get("FD_MODEL_CONFIG_PATH", None):
            self.model_config_path = env.get("FD_MODEL_CONFIG_PATH")

        # distributed config
        self.distributed_config_path = os.path.join(self.model_dir, "rank_mapping.csv")
        if os.getenv("DISTRIBUTED_CONFIG", None):
            self.distributed_config_path = os.getenv("DISTRIBUTED_CONFIG")

        # device config
        self.device = env.get("DEVICE", "GPU")
        self.device_ids = ",".join([str(i) for i in range(self.mp_num)])
        if self.device == "GPU":
            self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES",
                                        self.device_ids)
        else:
            raise Exception(f"unsupported device type: {self.device}")

        # Triton config
        self.max_prefill_batch = int(os.getenv("MAX_PREFILL_BATCH", 1))
        if self.max_prefill_batch <= 0:
            raise Exception(f"MAX_PREFILL_BATCH ({self.max_prefill_batch}) must be greater than 0")
        self.disable_streaming = int(os.getenv("DISABLE_STREAMING", 0))

        # max cached task num
        self.max_cached_task_num = int(os.getenv("MAX_CACHED_TASK_NUM", "128"))
        # if PUSH_MODE_HTTP_PORT is not configured, only GRPC service is enabled
        self.push_mode_http_port = int(os.getenv("PUSH_MODE_HTTP_PORT", "-1"))
        if self.push_mode_http_port > 0:
            grpc_port = os.getenv("GRPC_PORT", None)
            if grpc_port is None:
                raise Exception("GRPC_PORT cannot be None, while PUSH_MODE_HTTP_PORT>0")
            self.grpc_port = int(grpc_port)

        # http worker num
        self.push_mode_http_workers = int(os.getenv("PUSH_MODE_HTTP_WORKERS", "1"))
        if self.push_mode_http_workers < 1:
            raise Exception(f"PUSH_MODE_HTTP_WORKERS ({self.push_mode_http_workers}) must be positive")

        # Padlle commit id
        import paddle
        self.paddle_commit_id = paddle.version.commit

        # time interval for detecting whether the engine loop is normal during probing
        self.check_health_interval = int(os.getenv("CHECK_HEALTH_INTERVAL", 10))

        # model config
        self.dtype = env.get("DTYPE", "bfloat16")
        self.block_size = int(env.get("BLOCK_SIZE", 64))
        self.use_cache_kv_int8 = int(os.getenv("USE_CACHE_KV_INT8", 0))
        self.use_cache_kv_int4 = int(os.getenv("USE_CACHE_KV_INT4", 0))

        # infer config
        self.max_batch_size = int(env.get("BATCH_SIZE", 50))
        self.max_seq_len = int(env.get("MAX_SEQ_LEN", 8192))
        self.max_dec_len = int(env.get("MAX_DEC_LEN", 1024))
        self.enc_dec_block_num = int(os.getenv("ENC_DEC_BLOCK_NUM", 2))
        self.block_bs = float(env.get("BLOCK_BS", 50))
        self.block_ratio = float(os.getenv("BLOCK_RATIO", 0.75))
        self.bad_tokens = str(env.get("BAD_TOKENS", "-1"))
        self.first_token_id = int(os.getenv("FIRST_TOKEN_ID", 1))

        # infer queue port
        self.infer_port = int(os.getenv("INFER_QUEUE_PORT", 56666))

        # whether to use custom health checker
        self.use_custom_health_checker = int(os.getenv("USE_CUSTOM_HEALTH_CHECKER", 1))

        # Check the legality of requests
        self.seq_len_limit = int(env.get("MAX_SEQ_LEN", 8192))
        self.dec_len_limit = int(env.get("MAX_DEC_LEN", 1024))

        # warmup
        self.use_warmup = int(os.getenv("USE_WARMUP", 0)) == 1

        # uuid
        self.shm_uuid = os.getenv("SHM_UUID", '')

        # use huggingface tokenizer
        self.use_hf_tokenizer = int(os.getenv("USE_HF_TOKENIZER", 0)) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
        except:
            model_server_logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        self.read_from_config()
        self.postprocess()
        self.check()

    def postprocess(self):
        """
        calculate some parameters
        """
        if self.block_ratio >= 1.0:
            self.enc_dec_block_num = (self.max_dec_len + self.block_size - 1) // self.block_size
        self.max_query_block_num = (max(self.max_dec_len, self.max_seq_len) +
                               self.block_size - 1) // self.block_size
        self.max_query_block_num = (self.max_dec_len + self.max_seq_len +
                                self.block_size - 1) // self.block_size
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        self.total_block_num = int(self.block_bs * self.max_query_block_num)
        self.max_block_num = int(self.total_block_num * self.block_ratio)
        model_server_logger.info(f"max_block_num:{self.max_block_num}")

    def check(self):
        """
        check the legality of config
        """
        assert self.max_batch_size <= 256, (
            "The parameter `max_batch_size` is not allowed to exceed 256, "
            "but now it's {}.".format(self.max_batch_size)
        )
        assert self.seq_len_limit <= self.max_seq_len, (
            f"The seq_len_limit shouldn't greater than max_seq_len in model, "
            f"which means the exported MAX_SEQ_LEN should less than "
            f"{self.max_seq_len}, but now it's {self.seq_len_limit}."
        )
        assert self.dec_len_limit <= self.max_seq_len, (
            f"The dec_len_limit shouldn't greater than max_seq_len in model, "
            f"which means the exported MAX_DEC_LEN should less than "
            f"{self.max_seq_len}, but now it's {self.dec_len_limit}."
        )

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        model_server_logger.info(
            "=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    model_server_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            else:
                model_server_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        model_server_logger.info(
            "=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()

    def get_model_config(self):
        """
        load config file

        Returns:
            dict: the config file
        """
        model_config_json = json.load(open(self.model_config_path, 'r', encoding='utf-8'))
        return model_config_json

    def get_speculate_config(self):
        """
        get speculate_decoding related config

        Returns:
            SpeculateConfig: the speculate related config
        """
        speculate_config = SpeculateConfig()
        model_cfg = self.get_model_config()
        if model_cfg.get("speculate_method", "None") != "None":
            speculate_config.speculate_method = str(model_cfg["speculate_method"])
            speculate_config.speculate_max_draft_token_num = model_cfg[
                "speculate_max_draft_token_num"]
            speculate_config.speculate_max_ngram_size = model_cfg[
                "speculate_max_ngram_size"]

        if speculate_config.speculate_method not in ["None", "inference_with_reference"]:
            model_server_logger.error(f"Unsupport speculate method: {speculate_config.speculate_method}")

        return speculate_config

    def read_from_config(self):
        """
        reset model config from json file
        """
        from server.utils import get_logger
        logger = get_logger("model_server", "infer_config.log")
        config = self.get_model_config()

        def reset_value(self, value_name, key, config):
            if key in config:
                value = config[key]
                setattr(self, value_name, value)
                logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self, "block_size", "infer_model_block_size", config)
        reset_value(self, "max_seq_len", "infer_model_max_seq_len", config)

        assert self.seq_len_limit <= self.max_seq_len, f"The loading model requires len(input_ids) <= {self.max_seq_len}, but now the setting MAX_SEQ_LEN={self.seq_len_limit}."
        assert self.dec_len_limit <= self.max_seq_len, f"The loading model requires MAX_DEC_LEN <= {self.max_seq_len}, but now the setting MAX_DEC_LEN={self.dec_len_limit}."

    def get_unique_name(self, name):
        """
        get unique name

        Args:
            name (str): the name add uuid
        """
        return name + f"_{self.shm_uuid}"

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)


@dataclass
class SpeculateConfig:
    speculate_method: str = "None"
    speculate_max_draft_token_num: int = 1
    speculate_max_ngram_size: int = 1