# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import faiss
import numpy as np
import paddle
import paddle.distributed as dist
from tqdm import tqdm

from paddlenlp.transformers import AutoConfig, AutoModel, AutoTokenizer


class Clean_Query:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        input_data_path,
        output_data_path,
        template="{text}",
        dimension=1024,
        max_src_len=8192,
        normalize=True,
        dtype=None,
        similarity_threshold=0.75,
    ):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            truncation_side="right",
        )

        self.config = AutoConfig.from_pretrained(model_path)
        self.config.embedding_negatives_cross_device = False
        self.dtype = dtype if dtype else self.config.dtype

        # Initialize the distributed environment
        dist.init_parallel_env()
        world_size = dist.get_world_size()
        if world_size > 1:
            print(f"Running in multi-GPU mode with {world_size} GPUs.")
        else:
            print("Running in single-GPU or CPU mode.")

        # Initialize the embedding model
        self.model = AutoModel.from_pretrained(
            model_path, config=self.config, dtype=self.dtype, low_cpu_mem_usage=False
        )
        self.model.eval()

        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.template = template
        self.dimension = dimension
        self.max_src_len = max_src_len
        self.normalize = normalize
        self.similarity_threshold = similarity_threshold

    def _preprocess(self, texts):
        """Pre-process inputs."""
        template_prefix, template_suffix = self.template.split("{text}")
        prefix_tokens = self.tokenizer(template_prefix, add_special_tokens=False).input_ids
        suffix_tokens = self.tokenizer(template_suffix, add_special_tokens=False).input_ids

        # If the template does not contain a suffix token, add the EOS token
        if len(suffix_tokens) == 0:
            suffix_tokens = [self.tokenizer.eos_token_id]
        # If the template does not contain a prefix token, add the BOS token
        if len(prefix_tokens) == 0:
            prefix_tokens = [self.tokenizer.bos_token_id]

        available_len = self.max_src_len - len(prefix_tokens) - len(suffix_tokens)
        truncated_token_ids = self._batch_truncate_and_tokenize(texts, available_len)
        complete_token_ids = [prefix_tokens + tid + suffix_tokens for tid in truncated_token_ids]
        position_ids = [list(range(len(cid))) for cid in complete_token_ids]
        max_len = max([len(cid) for cid in complete_token_ids])
        embedding_indices = [[idx, len(cid) - 1] for idx, cid in enumerate(complete_token_ids)]

        inputs = self.tokenizer.pad(
            {
                "input_ids": complete_token_ids,
                "position_ids": position_ids,
                "embedding_indices": embedding_indices,
            },
            padding="max_length",
            return_attention_mask=True,
            max_length=max_len,
            return_tensors="pd",
        )
        return inputs

    def _batch_truncate_and_tokenize(self, texts, available_len):
        """Tokenize the batch of texts."""
        batch_tokenized = self.tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=True, max_length=available_len
        )

        truncated_token_ids = [token_ids for token_ids in batch_tokenized["input_ids"]]
        return truncated_token_ids

    def _forward(self, inputs, dimension):
        """Run model forward."""
        input_type = type(inputs["input_ids"])
        outputs = self.model(**inputs)
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]
        last_hidden_state = hidden_states[:, 0]

        if dimension > self.config.hidden_size:
            raise ValueError(
                f"Dimension ({dimension}) cannot be greater than hidden_size ({self.config.hidden_size})."
            )
        elif dimension != self.config.hidden_size:
            last_hidden_state = last_hidden_state[:, :dimension]

        if self.normalize:
            last_hidden_state = paddle.nn.functional.normalize(last_hidden_state, axis=-1)

        last_hidden_state = last_hidden_state.astype("float16").tolist()
        return last_hidden_state

    @paddle.no_grad()
    def get_embedding(self, texts, dimension=None):
        """Get inference sequence."""
        if dimension is None:
            dimension = self.dimension
        inputs = self._preprocess(texts)
        if self.config.model_type in ["xlm-roberta"]:
            del inputs["embedding_indices"]
            del inputs["position_ids"]
        outputs = self._forward(inputs, dimension)
        return outputs

    def clean(self):
        data_list = []
        with open(self.input_data_path, "r") as f:
            for line in tqdm(f):
                data_list.append(json.loads(line))

        query_list = [single_data["query"] for single_data in data_list]

        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        chunk_size = len(query_list) // world_size
        if rank == world_size - 1:
            # The last process handles the remaining data
            query_data_chunk = query_list[rank * chunk_size :]
        else:
            query_data_chunk = query_list[rank * chunk_size : (rank + 1) * chunk_size]

        batch_size = 4  # Adjust batch size according to your hardware and needs
        local_q_vecs = []

        # Use tqdm to iterate over query_data_chunk and get embeddings in batches
        for batch in tqdm(range(0, len(query_data_chunk), batch_size), desc="Processing query embeddings"):
            batch_start = batch
            batch_end = min(batch_start + batch_size, len(query_data_chunk))
            batch_texts = query_data_chunk[batch_start:batch_end]

            # Call get_embedding to obtain embeddings for the current batch
            batch_embeddings = self.get_embedding(batch_texts)
            local_q_vecs.extend(batch_embeddings)

        local_q_vecs_file = f"local_q_vecs_rank_{rank}.npy"
        np.save(local_q_vecs_file, local_q_vecs)
        dist.barrier()  # Ensure all cards have reached this point before continuing

        if rank == 0:
            all_q_vecs_list = []
            world_size = paddle.distributed.get_world_size()

            for i in range(world_size):
                local_q_vecs_file = f"local_q_vecs_rank_{i}.npy"

                # Load the embedding vector file from each process
                local_q_vecs = np.load(local_q_vecs_file)
                all_q_vecs_list.append(local_q_vecs)

            all_q_vecs = []
            for q_vecs in all_q_vecs_list:
                all_q_vecs.extend(q_vecs)
            q_vecs = np.asarray(all_q_vecs, dtype=np.float32)

            index = faiss.IndexFlatIP(self.dimension)
            if paddle.is_compiled_with_cuda():
                co = faiss.GpuMultipleClonerOptions()
                co.shard = False
                co.useFloat16 = False
                index = faiss.index_cpu_to_all_gpus(index, co=co)

            temp_query_embedding = q_vecs[0].reshape(1, -1)
            # faiss.normalize_L2(temp_query_embedding)
            # print(q_vecs.shape)
            # print(temp_query_embedding)
            # print(temp_query_embedding.shape)
            index.add(temp_query_embedding)

            clean_data_list = [data_list[0]]
            temp_query_list = [query_list[0]]
            for i in tqdm(range(1, len(query_list))):
                single_query_embedding = q_vecs[i].reshape(1, -1)
                # faiss.normalize_L2(single_query_embedding)

                if i < 3:
                    top_values, top_indices = index.search(single_query_embedding, 1)
                else:
                    top_values, top_indices = index.search(single_query_embedding, 3)

                if top_values[0][0] < self.similarity_threshold:
                    clean_data_list.append(data_list[i])
                    index.add(single_query_embedding)
                    temp_query_list.append(query_list[i])
                # else:
                #     print(query_list[i])
                #     for j in range(top_values.shape[1]):
                #         print(f"similarity:{top_values[0][j]} query:{temp_query_list[top_indices[0][j]]}")
                #     print('********************************')
                #     continue
                # if i%10000==0:
                #     print(len(clean_data_list))

            with open(self.output_data_path, "w", encoding="utf-8") as f:
                for data in clean_data_list:
                    f.write(json.dumps(data, ensure_ascii=False))
                    f.write("\n")


if __name__ == "__main__":
    model_path = "BAAI/bge-m3"
    tokenizer_path = "BAAI/bge-m3"
    input_data_path = "./toy_data/toy_source.json"
    output_data_path = "./toy_data/test_clean.json"
    test_clean = Clean_Query(
        model_path,
        tokenizer_path,
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        similarity_threshold=0.70,
    )
    test_clean.clean()
