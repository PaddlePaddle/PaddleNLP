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
""" Run embedding server. """

import argparse
import base64
import threading
from typing import List, Optional

import numpy as np
import paddle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from paddlenlp.transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    ChatGLMTokenizer,
    ChatGLMv2Tokenizer,
)
from paddlenlp.trl import llm_utils


class Request(BaseModel):
    """Request"""

    input: List[str]
    model: Optional[str] = None
    encoding_format: Optional[str] = None


class Response(BaseModel):
    """Response"""

    error_code: Optional[int] = 0
    error_msg: Optional[str] = "Success"
    data: Optional[list] = None
    model: Optional[str] = None
    object: Optional[str] = "list"
    usage: Optional[dict] = None


def setup_args():
    """setup_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The directory of model.")
    parser.add_argument("--dimension", type=int, default=768, help="Default parameter for embedding inference.")
    parser.add_argument("--dtype", type=str, help="Specify the data type for model computation.")
    parser.add_argument("--max_src_len", type=int, default=3072, help="The max length of src.")
    parser.add_argument("--port", type=int, default=8000, help="The port of embedding server.")
    return parser.parse_args()


class Predictor:
    """Predictor"""

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            truncation_side="right",
        )
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)

        tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

        self.dtype = args.dtype if args.dtype else self.config.dtype

        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            dtype=self.dtype,
            tensor_parallel_output=False,
            low_cpu_mem_usage=False,
        )
        self.model.eval()

    def _preprocess(self, source):
        if self.tokenizer.chat_template is not None:
            source = [source] if isinstance(source, str) else source
            source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.args.max_src_len,
            truncation=True,
            return_position_ids=True if not isinstance(self.tokenizer, ChatGLMTokenizer) else False,
            truncation_side="left",
            return_tensors="pd",
            padding=True,
            # when use chat_template, it should not add special tokens
            # chatglm2 prefix-tokens can not be tokenized into ids
            add_special_tokens=self.tokenizer.chat_template is None
            or isinstance(self.tokenizer, (ChatGLMv2Tokenizer, ChatGLMTokenizer)),
        )
        return tokenized_source

    def _forward(self, inputs, dimension):
        """Run model forward."""
        last_hidden_state = self.model(**inputs)[1]

        if dimension > self.config.hidden_size:
            raise ValueError(
                f"Dimension ({dimension}) cannot be greater than hidden_size ({self.config.hidden_size})."
            )
        elif dimension != self.config.hidden_size:
            last_hidden_state = paddle.nn.functional.normalize(last_hidden_state[:, :dimension], axis=-1)

        last_hidden_state = last_hidden_state.astype("float16").tolist()
        return last_hidden_state

    @paddle.no_grad()
    def __call__(self, texts, dimension=None):
        """Get inference sequence."""
        if dimension is None:
            dimension = self.args.dimension
        inputs = self._preprocess(texts)
        outputs = self._forward(inputs, dimension)
        return outputs, sum([len(input_ids) for input_ids in inputs["input_ids"]])


def create_app(args):
    """create_app"""
    app = FastAPI()
    predictor = Predictor(args)
    lock = threading.Lock()

    @app.post("/v1/embeddings")
    def get_embeddings(req: Request) -> Response:
        with lock:
            try:
                # [batch_size, embedding_dim]
                embeddings, total_tokens = predictor(req.input, predictor.args.dimension)

                if req.encoding_format == "base64":
                    data = [
                        {
                            "embedding": base64.b64encode(np.array(embedding).tobytes()).decode("utf-8"),
                            "index": i,
                            "object": "embedding",
                        }  # 将numpy数组转换为列表
                        for i, embedding in enumerate(embeddings)
                    ]
                else:
                    data = [
                        {"embedding": embedding, "index": i, "object": "embedding"}  # 将numpy数组转换为列表
                        for i, embedding in enumerate(embeddings)
                    ]

                usage = {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                }

                return Response(data=data, model=args.model_name_or_path, object="list", usage=usage)
            except Exception as err:
                return Response(
                    data=[],
                    model=args.model_name_or_path,
                    object="list",
                    error_code=1000,
                    error_msg=f"Error type: {type(err).__name__}, Error message: {err!s}",
                )

    return app


if __name__ == "__main__":
    args = setup_args()
    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="debug")


# 使用方式
# 服务启动
# python flash_server_embed.py --model_name_or_path intfloat/e5-base-v2

# 服务调用参考test_flask_server_embed.py
