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

import paddle
import paddle.nn.functional as F
from arguments import DataArguments, ModelArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from modeling import BloomBiEncoderModel

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = BloomBiEncoderModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype="bfloat16",
        low_cpu_mem_usage=True,
        normalized=model_args.normalized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_flash_attention=model_args.use_flash_attention,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model.eval()
    query = ["Five women walk along a beach wearing flip-flops"]
    passage = [
        "Some women with flip-flops on, are walking along the beach",
        "The 4 women are sitting on the beach.",
        "There was a reform in 1996.",
    ]
    with paddle.no_grad():
        decoder_inputs = tokenizer(query, padding=True, return_tensors="pd")
        query_embedding = model.encode(decoder_inputs)
        logger.info(f"Query embeddings {query_embedding}")
        decoder_inputs = tokenizer(
            passage,
            padding=True,
            return_tensors="pd",
        )
        passage_embedding = model.encode(decoder_inputs)
        logger.info(f"Passage embeddings {passage_embedding}")

        probs = F.cosine_similarity(query_embedding, passage_embedding)
        print(probs)


if __name__ == "__main__":
    main()
