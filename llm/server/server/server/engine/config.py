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
from datetime import datetime
import re

from server.utils import model_server_logger
from server.download_model import download_from_txt
from paddlenlp.experimental.transformers import SpeculateArgument
from paddlenlp.generation import GenerationConfig


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
        self.model_dir = env.get("MODEL_DIR", "/opt/output/Serving/models")
        if not self.model_dir:
            raise Exception("The parameter MODEL_DIR is None.")
        self.mp_num = int(env.get("MP_NUM", 8))
        self.config_json_file = env.get("CONFIG_JSON_FILE", "config.json")
        self.model_config_path = os.path.join(self.model_dir, self.config_json_file)

        # device config
        self.device = env.get("DEVICE", "GPU")
        self.device_ids = ",".join([str(i) for i in range(self.mp_num)])
        if self.device == "GPU":
            self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.device_ids)
        else:
            raise Exception(f"unsupported device type: {self.device}")

        # multi-node config
        self.nnode = int(env.get("MP_NNODE", "1"))
        assert self.mp_num % self.nnode == 0, f"mp_num: {self.mp_num} should be divisible by nnode: {self.nnode}"
        self.mp_num_per_node = self.mp_num // self.nnode
        self.host_ip = os.getenv("HOST_IP", "127.0.0.1")
        if self.nnode > 1:
            self.ips = os.getenv("POD_IPS")

        # Triton config
        self.max_prefill_batch = int(os.getenv("MAX_PREFILL_BATCH", 1))
        if self.max_prefill_batch <= 0:
            raise Exception(f"MAX_PREFILL_BATCH ({self.max_prefill_batch}) must be greater than 0")
        self.disable_streaming = int(os.getenv("DISABLE_STREAMING", 0))

        # max cached task num
        self.max_cached_task_num = int(os.getenv("MAX_CACHED_TASK_NUM", "128"))
        # if SERVICE_HTTP_PORT is not configured, only GRPC service is enabled
        self.push_mode_http_port = int(os.getenv("SERVICE_HTTP_PORT", "-1"))
        if self.push_mode_http_port > 0:
            grpc_port = os.getenv("SERVICE_GRPC_PORT", None)
            if grpc_port is None:
                raise Exception("SERVICE_GRPC_PORT cannot be None, while SERVICE_HTTP_PORT>0")
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
        self.return_full_hidden_states = int(os.getenv("RETURN_FULL_HIDDEN_STATES", 0))

        # infer queue port
        self.infer_port = int(os.getenv("INTER_PROC_PORT", 56666))

        # whether to use custom health checker
        self.use_custom_health_checker = int(os.getenv("USE_CUSTOM_HEALTH_CHECKER", 1))

        # Check the legality of requests
        self.seq_len_limit = int(env.get("MAX_SEQ_LEN", 8192))
        self.dec_len_limit = int(env.get("MAX_DEC_LEN", 1024))

        # warmup
        self.use_warmup = int(os.getenv("USE_WARMUP", 0)) == 1

        # uuid
        self.shm_uuid = os.getenv("SHM_UUID", "")

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
        self.max_query_block_num = (max(self.max_dec_len, self.max_seq_len) + self.block_size - 1) // self.block_size
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        self.total_block_num = int(self.block_bs * self.max_query_block_num)
        self.max_block_num = int(self.total_block_num * self.block_ratio)
        model_server_logger.info(f"max_block_num:{self.max_block_num}")

    def check(self):
        """
        check the legality of config
        """
        import math

        assert (
            self.max_batch_size <= 256
        ), "The parameter `max_batch_size` is not allowed to exceed 256, " "but now it's {}.".format(
            self.max_batch_size
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
        if os.getenv("DISABLE_CAPACITY_CHECKER", "0") == 1:
            # max_output_token_num
            max_output_token_num = (
                self.total_block_num - self.max_block_num
            ) * self.block_size + self.enc_dec_block_num * self.block_size
            assert max_output_token_num >= self.dec_len_limit, (
                f"The available output token number of the service is {max_output_token_num}, "
                f"which is less than the setting MAX_DEC_LEN:{self.dec_len_limit}. "
            )

            # Maximum input length of a single query that the service can handle
            max_input_token_num = int(math.floor(self.max_block_num * self.block_size - self.dec_token_num))
            assert max_input_token_num >= self.seq_len_limit, (
                f"The available input token number of the service is {max_input_token_num}, "
                f"which is less than the setting MAX_SEQ_LEN:{self.seq_len_limit}. "
            )

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        model_server_logger.info("=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    model_server_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            else:
                model_server_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        model_server_logger.info("=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()


    def _get_download_model(self, model_type="default"):
        env = os.environ
        model_name = env.get("model_name")
        # Define supported model patterns
        supported_patterns = [
            r".*Qwen.*", 
            r".+Llama.+",
            r".+Mixtral.+", 
            r".+DeepSeek.+",
        ]
        
        # Check if model_name matches any supported pattern
        if not any(re.match(pattern, model_name) for pattern in supported_patterns):
            raise ValueError(
                f"{model_name} is not in the supported list. Currently supported models: Qwen, Llama, Mixtral, DeepSeek. Please check the model name from this document https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md"
            )
        model_server_logger.info(f"Start downloading model: {model_name}")
        tag=env.get("tag")
        base_url=f"https://paddlenlp.bj.bcebos.com/models/static/{tag}/{model_name}"
        if self.nnode == 1:
            # single node model
            temp_file = "model"
        elif env.get("POD_0_IP", "127.0.0.1") == self.host_ip:
            # Master node model
            temp_file = "node1"
        else:
            temp_file = "node2"
        if model_type == "default":
            path = self.model_dir 
        elif model_type == "speculate":
            path = os.getenv("SPECULATE_MODEL_PATH")
            temp_file="mtp"
        model_url = base_url+f"/{temp_file}"
        download_from_txt(model_url, path)



    def get_model_config(self):
        """
        load config file

        Returns:
            dict: the config file
        """
        
        model_config_json = json.load(open(self.model_config_path, "r", encoding="utf-8"))
        return model_config_json

    def get_speculate_config(self):
        """
        get speculate_decoding related config

        Returns:
            SpeculateArgument: the speculate related arguments
        """
        from server.utils import get_logger

        model_cfg = self.get_model_config()

        if model_cfg.get("speculate_method", "None") != "None":
            speculate_method = os.getenv("SPECULATE_METHOD", model_cfg["speculate_method"])
            speculate_max_draft_token_num = int(
                os.getenv("SPECULATE_MAX_DRAFT_TOKEN_NUM", model_cfg["speculate_max_draft_token_num"])
            )
            speculate_model_name_or_path = os.getenv("SPECULATE_MODEL_PATH", None)
            speculate_model_quant_type = os.getenv("SPECULATE_MODEL_QUANT_TYPE", "weight_only_int8")
            speculate_max_ngram_size = int(
                os.getenv("SPECULATE_MAX_NGRAM_SIZE", model_cfg["speculate_max_ngram_size"])
            )

            if speculate_method in ["eagle", "mtp"]:
                assert (
                    speculate_model_name_or_path is not None
                ), "[eagle, mtp] method must be set by env SPECULATE_MODEL_PATH"

            if not os.path.exists(speculate_model_name_or_path):
                self._get_download_model(model_type="speculate")
            
            speculate_args = SpeculateArgument.build_from_serving(
                speculate_method=speculate_method,
                speculate_max_draft_token_num=speculate_max_draft_token_num,
                speculate_max_ngram_size=speculate_max_ngram_size,
                model_name_or_path=speculate_model_name_or_path,
                quant_type=speculate_model_quant_type,
                max_batch_size=self.max_batch_size,
                total_max_length=self.max_seq_len,
                max_length=self.max_dec_len,
                dtype=self.dtype,
                mla_use_matrix_absorption=model_cfg.get("mla_use_matrix_absorption", False),
            )

            logger = get_logger("model_server", "infer_config.log")
            logger.info(f"Speculate info: {speculate_args}")
            return speculate_args
        else:
            return SpeculateArgument.build_from_serving(speculate_method="None")

    def read_from_config(self):
        """
        reset model config from json file
        """
        from server.utils import get_logger

        logger = get_logger("model_server", "infer_config.log")
        env = os.environ
        model_name=env.get("model_name")
        if model_name:
            self._get_download_model()

        config = self.get_model_config()
        # check paddle nlp version
        tag = os.getenv("tag")
        if tag not in config["paddlenlp_version"]:
            logger.warning(f"Current image paddlenlp version {tag} doesn't match the model paddlenlp version {config['paddlenlp_version']} ")

        def reset_value(self, value_name, key, config):
            if key in config:
                value = config[key]
                setattr(self, value_name, value)
                logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self, "block_size", "infer_model_block_size", config)
        reset_value(self, "max_seq_len", "infer_model_max_seq_len", config)
        reset_value(self, "return_full_hidden_states", "return_full_hidden_states", config)
        reset_value(self, "dtype", "infer_model_dtype", config)
        reset_value(self, "use_cache_kv_int8", "infer_model_cachekv_int8_type", config)
        if self.use_cache_kv_int8 == None:
            self.use_cache_kv_int8 = 0
        else:
            self.use_cache_kv_int8 = 1
        if self.seq_len_limit > self.max_seq_len:
            self.seq_len_limit = self.max_seq_len
            logger.warning(f"The loading model requires len(input_ids) <= {self.max_seq_len}, now reset MAX_SEQ_LEN.")

        if self.dec_len_limit > self.max_seq_len:
            self.dec_len_limit = self.max_seq_len
            logger.warning(f"The loading model requires MAX_DEC_LEN <= {self.max_seq_len}, now reset MAX_DEC_LEN.")

    def get_unique_name(self, name):
        """
        get unique name

        Args:
            name (str): the name add uuid
        """
        return name + f"_{self.shm_uuid}"

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)

global_config = Config()