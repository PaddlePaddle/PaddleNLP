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

import dataclasses
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
from paddle.distributed import fleet
from tqdm import tqdm

from paddlenlp.transformers import AutoConfig
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.predict.predictor import (
    DygraphBlockInferencePredictor,
    ModelArgument,
    PredictorArgument,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.transformers import (
    AutoInferenceModelForCausalLM,
    Llama3Tokenizer,
    LlamaTokenizer,
)

MODEL_FLAG = ""
MAX_SEQ_LENGTH = 0
QUERY_DOC_FLAG_FOR_LLARA = ""


class DygraphBlockInferenceHiddenPredictor(DygraphBlockInferencePredictor):
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        super().__init__(config, tokenizer, model, **kwargs)

    @paddle.no_grad()
    def encode(self, sentences: list[str]):

        if MODEL_FLAG == "llara":
            logger.warning('MODEL_FLAG == "llara"')
            sentences = self.preprocess_sentences_for_llara(sentences, QUERY_DOC_FLAG_FOR_LLARA)

        total = 0
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.config.batch_size), desc="Batches"):
            sentences_batch = sentences[start_index : start_index + self.config.batch_size]

            self._preprocess(sentences_batch)
            if self.proposer is not None:
                self.proposer.insert_query(
                    base_model_inputs=self.model_inputs, real_bs=len(sentences_batch), seq_lens=self.seq_lens
                )

            if self.proposer is not None:
                self.proposer.run(
                    self.model_inputs,
                    # real_batch_size=self.batch_size,
                    real_batch_size=len(sentences_batch),
                    seq_lens_this_time=self.model_inputs["seq_lens_this_time"],
                    base_model_full_hidden_states=self.full_hidden_states,
                )

            inputs = self.model_inputs
            
            _, full_hidden_states = self.model(
                input_ids=inputs["input_ids"],
                seq_lens_this_time=inputs["seq_lens_this_time"],
                caches=inputs["cache_kvs"],
                seq_lens_encoder=inputs["seq_lens_encoder"],
                seq_lens_decoder=inputs["seq_lens_decoder"],
                block_tables=inputs["block_tables"],
                rope_emb=inputs["rope_emb"],
                kv_cache_reuse=self.config.kv_cache_reuse,
            )

            last_hidden_state_tensor = self.split_hidden_states_by_seq_lens(
                full_hidden_states, inputs["seq_lens_this_time"]
            )
            total += last_hidden_state_tensor.shape[0]

            assert last_hidden_state_tensor.shape[0] == len(
                sentences_batch
            ), f"Output batch size mismatch: {last_hidden_state_tensor.shape[0]} vs {len(sentences_batch)}"
            assert (
                last_hidden_state_tensor.shape[1] == self.model.config.hidden_size
            ), f"Hidden size mismatch: {last_hidden_state_tensor.shape[1]} vs {self.model.config.hidden_size}"

            if self.config.normalized:
                embeddings = paddle.nn.functional.normalize(last_hidden_state_tensor, p=2, axis=-1)

            all_embeddings.append(embeddings.cpu().numpy().astype("float32"))

        return np.concatenate(all_embeddings, axis=0)

    def split_hidden_states_by_seq_lens(self, hidden_states, seq_lens_this_time):
        """
        Args:
            hidden_states (Tensor): shape [total_seq_len, hidden_size], e.g. [135, 2048]
            seq_lens_this_time (Tensor): shape [batch_size, 1], e.g. [[127], [8]]

        Returns:
            Tensor: shape [batch_size, hidden_size]
        """
        if hasattr(seq_lens_this_time, "numpy"):  # Paddle tensor
            seq_lens = seq_lens_this_time.numpy().flatten().tolist()
        else:
            seq_lens = [x[0] if isinstance(x, list) else x for x in seq_lens_this_time]

        if self.config.sentence_pooling_method == "last":
            if self.config.tokenizer.padding_side == "right":
                split_hidden_states = []
                start = 0
                for length in seq_lens:
                    end = start + length - 1
                    split_hidden_states.append(hidden_states[end])
                    start = start + length

        elif self.config.sentence_pooling_method == "last_8":
            split_hidden_states = []
            start = 0
            for length in seq_lens:
                end = start + length - 1
                split_hidden_states.append(paddle.mean(hidden_states[end - 7 : end + 1], axis=0))
                start = start + length

        else:
            raise f"the sentence_pooling_method {self.config.sentence_pooling_method} is not supported"
        return paddle.stack(split_hidden_states, axis=0)  # shape: [batch_size, hidden_size]

    def preprocess_sentences_for_llara(self, sentences: List[str], query_or_doc: str, **kwargs) -> List[str]:

        prefix = '"'
        if query_or_doc == "query":
            suffix = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
        elif query_or_doc == "doc":
            suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
        else:
            raise ValueError(f"Invalid query_or_doc: {query_or_doc}")

        logger.warning(f"query_or_doc: {query_or_doc}")

        sentences_after_process = []
        import tqdm

        for sentence in tqdm.tqdm(sentences, desc="preprocess_sentences_for_llara"):
            inputs = self.tokenizer(
                sentence,
                return_tensors=None,
                max_length=MAX_SEQ_LENGTH - 20,
                truncation=True,
                add_special_tokens=False,
            )
            sentences_after_process.append(self.tokenizer.decode(inputs["input_ids"], skip_special_tokens=True))

        sentences_after_process = [prefix + " " + sentence + " " + suffix for sentence in sentences_after_process]

        return sentences_after_process


class HiddenPredictorWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        normalized: bool = True,
        sentence_pooling_method: str = "last",
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        tokenizer=None,
        eval_batch_size: int = 32,
        max_seq_length: int = 512,
        model_flag: str = None,
        dtype: str = "float32",
        quant_type: str = None,
        kv_cache_reuse: bool = False,
    ):
        self.predictor_args = PredictorArgument()
        self.model_args = ModelArgument()

        override_fields = {
            "model_name_or_path": model_name_or_path,
            "sentence_pooling_method": sentence_pooling_method,
            "dtype": dtype,
            "quant_type": quant_type,
            "return_full_hidden_states": 1,
            "inference_model": True,
            "block_attn": True,
            "batch_size": eval_batch_size,
            "kv_cache_reuse": bool(kv_cache_reuse),
        }
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.model_flag = model_flag
        self.quant_type = quant_type
        self.tokenizer = tokenizer

        for field in dataclasses.fields(self.predictor_args):
            if field.name in override_fields and override_fields[field.name] is not None:
                setattr(self.predictor_args, field.name, override_fields[field.name])

        for field in dataclasses.fields(self.model_args):
            if field.name in override_fields and override_fields[field.name] is not None:
                setattr(self.model_args, field.name, override_fields[field.name])

        self.predictor_args.tokenizer = self.tokenizer
        self.predictor_args.sentence_pooling_method = self.sentence_pooling_method
        self.predictor_args.normalized = self.normalized
        self.predictor = self._create_predictor()

    def _create_predictor(self):

        model_config = AutoConfig.from_pretrained(self.predictor_args.model_name_or_path)

        llm_utils.set_triton_cache(self.predictor_args.model_name_or_path, self.predictor_args.mode)
        try:
            from paddle.utils import try_import

            try_import("paddlenlp_ops")
        except ImportError:
            logger.warning("paddlenlp_ops does not exist, please install paddlenlp_ops.")
            #return
        tensor_parallel_degree = paddle.distributed.get_world_size()
        if tensor_parallel_degree > 1:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)

        paddle.set_device(self.predictor_args.device)
        paddle.set_default_dtype(self.predictor_args.dtype)
        from paddlenlp.utils.env import USE_FAST_TOKENIZER

        self.tokenizer.use_fast = USE_FAST_TOKENIZER

        # init chat_template for tokenizer
        llm_utils.init_chat_template(self.tokenizer, self.model_name_or_path, self.predictor_args.chat_template)
        tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()
        # TODO(wj-Mcat): fix llama tokenzier pad_token bug
        if (isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer))) and not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoInferenceModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            config=model_config,
            predictor_args=self.predictor_args,
            model_args=self.model_args,
            dtype=self.dtype,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
        )

        predictor_class_name = (
            "DygraphBlockInferenceHiddenPredictor"  # execute_mode + inference_mode + "Hidden" + "Predictor"
        )

        import_class = sys.modules[__name__]
        predictor_class = getattr(import_class, predictor_class_name)

        cache_kvs_shape = None  # used for not block_attn/append_attn
        cache_k_shapes = None  # used for block_attn/append_attn
        cache_v_shapes = None  # used for block_attn/append_attn

        predictor = predictor_class(
            self.predictor_args,
            tokenizer=self.tokenizer,
            model=model,
            cache_k_shapes=cache_k_shapes,
            cache_v_shapes=cache_v_shapes,
            cache_kvs_shape=cache_kvs_shape,
            model_args=self.model_args,
        )

        return predictor

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        This function will be used to encode queries for retrieval task
        if there is a instruction for queries, we will add it to the query text
        """

        global MODEL_FLAG
        global MAX_SEQ_LENGTH
        global QUERY_DOC_FLAG_FOR_LLARA
        MODEL_FLAG = self.model_flag
        MAX_SEQ_LENGTH = self.max_seq_length
        QUERY_DOC_FLAG_FOR_LLARA = "query"

        if self.query_instruction is not None:
            input_texts = [f"{self.query_instruction}{query}" for query in queries]
        else:
            input_texts = queries

        assert isinstance(input_texts, list), "input_texts should be a list"
        assert len(input_texts) == len(queries), f"Mismatch in number of queries: {len(input_texts)} vs {len(queries)}"

        encode_results = self.encode_sentences(input_texts=input_texts)

        assert isinstance(encode_results, np.ndarray), "encode_results should be a numpy array"
        assert encode_results.shape[0] >= len(
            input_texts
        ), f"Encoded query count mismatch: {encode_results.shape[0]} vs {len(input_texts)}"

        return encode_results[: len(input_texts)]

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        """
        This function will be used to encode corpus for retrieval task
        if there is a instruction for docs, we will add it to the doc text
        """

        global MODEL_FLAG
        global QUERY_DOC_FLAG_FOR_LLARA
        MODEL_FLAG = self.model_flag
        QUERY_DOC_FLAG_FOR_LLARA = "doc"

        if isinstance(corpus[0], dict):
            if self.document_instruction is not None:
                input_texts = [
                    "{}{} {}".format(self.document_instruction, doc.get("title", ""), doc["text"]).strip()
                    for doc in corpus
                ]
            else:
                input_texts = ["{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus]
        else:
            if self.document_instruction is not None:
                input_texts = [f"{self.document_instruction}{doc}" for doc in corpus]
            else:
                input_texts = corpus

        encode_results = self.encode_sentences(input_texts=input_texts)
        assert encode_results.shape[0] >= len(
            input_texts
        ), f"Encoded query count mismatch: {encode_results.shape[0]} vs {len(input_texts)}"

        return encode_results[: len(input_texts)]

    def encode_sentences(self, input_texts):
        encode_results = self.predictor.encode(input_texts)

        return encode_results
