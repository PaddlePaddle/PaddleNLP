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


class MiningNegativeSamples:
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
    ):
        # initialize the tokenizer
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

    def mining(self):
        query_pos_dict = {}
        query_data_list = []
        pos_data_list = []
        temp_data_list = []
        count = 0
        with open(self.input_data_path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                query = data["query"]
                pos_passage = data["pos_passage"][0]

                if query not in query_pos_dict:
                    temp_data_list.append(data)
                    query_data_list.append(query)
                    pos_data_list.append(pos_passage)
                    query_pos_dict[query] = [pos_passage]
                else:
                    # print('error1',query)
                    count += 1
                    query_pos_dict[query].append(pos_passage)

        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        assert len(pos_data_list) == len(query_data_list)
        chunk_size = len(pos_data_list) // world_size
        if rank == world_size - 1:
            # The last process handles the remaining data
            pos_data_chunk = pos_data_list[rank * chunk_size :]
            query_data_chunk = query_data_list[rank * chunk_size :]
        else:
            pos_data_chunk = pos_data_list[rank * chunk_size : (rank + 1) * chunk_size]
            query_data_chunk = query_data_list[rank * chunk_size : (rank + 1) * chunk_size]

        batch_size = 4  # Adjust batch size according to your hardware and needs
        local_p_vecs = []
        local_q_vecs = []

        # Use tqdm to iterate over query_data_chunk and get embeddings in batches
        for batch in tqdm(range(0, len(pos_data_chunk), batch_size), desc="Processing passage embeddings"):
            batch_start = batch
            batch_end = min(batch_start + batch_size, len(pos_data_chunk))
            batch_texts = pos_data_chunk[batch_start:batch_end]

            # Call get_embedding to obtain embeddings for the current batch
            batch_embeddings = self.get_embedding(batch_texts)
            local_p_vecs.extend(batch_embeddings)

        for batch in tqdm(range(0, len(query_data_chunk), batch_size), desc="Processing query embeddings"):
            batch_start = batch
            batch_end = min(batch_start + batch_size, len(query_data_chunk))
            batch_texts = query_data_chunk[batch_start:batch_end]

            batch_embeddings = self.get_embedding(batch_texts)
            local_q_vecs.extend(batch_embeddings)

        local_p_vecs_file = f"local_p_vecs_rank_{rank}.npy"
        local_q_vecs_file = f"local_q_vecs_rank_{rank}.npy"
        np.save(local_p_vecs_file, local_p_vecs)
        np.save(local_q_vecs_file, local_q_vecs)
        dist.barrier()  # Ensure all cards have reached this point before continuing

        if rank == 0:
            all_p_vecs_list = []
            all_q_vecs_list = []
            world_size = paddle.distributed.get_world_size()

            for i in range(world_size):
                local_p_vecs_file = f"local_p_vecs_rank_{i}.npy"
                local_q_vecs_file = f"local_q_vecs_rank_{i}.npy"

                # Load the embedding vector file from each process
                local_p_vecs = np.load(local_p_vecs_file)
                local_q_vecs = np.load(local_q_vecs_file)

                all_p_vecs_list.append(local_p_vecs)
                all_q_vecs_list.append(local_q_vecs)

            all_q_vecs = []
            for q_vecs in all_q_vecs_list:
                all_q_vecs.extend(q_vecs)
            q_vecs = np.asarray(all_q_vecs, dtype=np.float32)

            all_p_vecs = []
            for p_vecs in all_p_vecs_list:
                all_p_vecs.extend(p_vecs)
            p_vecs = np.asarray(all_p_vecs, dtype=np.float32)

            index = faiss.IndexFlatIP(len(p_vecs[0]))
            p_vecs = np.asarray(p_vecs, dtype=np.float32)
            if paddle.is_compiled_with_cuda():
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = False
                index = faiss.index_cpu_to_all_gpus(index, co=co)

            index.add(p_vecs)

            count = 0
            batch_size = 16
            output_data_list = []
            for i in tqdm(range(0, len(q_vecs), batch_size)):
                batch_queries = query_data_list[i : i + batch_size]
                batch_q_vecs = q_vecs[i : i + batch_size]
                _, batch_ids = index.search(batch_q_vecs, k=10)

                for j, ids in enumerate(batch_ids):
                    query = batch_queries[j]
                    converted = [id for id in ids]
                    neg_list = []

                    for id in converted:
                        if pos_data_list[id] in query_pos_dict[query]:
                            continue
                        neg_list.append(pos_data_list[id])

                    # you can mining k negatives,there k==2
                    if len(neg_list) > 2:
                        neg_list = neg_list[:2]
                        assert query == temp_data_list[i + j]["query"]
                        temp_data_list[i + j]["neg_passage"] = neg_list
                        output_data_list.append(temp_data_list[i + j])
                    else:
                        print("error2", query)
                        count += 1

            del index

            with open(self.output_data_path, "w", encoding="utf-8") as f:
                for item in tqdm(output_data_list):
                    f.write(json.dumps(item, ensure_ascii=False))
                    f.write("\n")


if __name__ == "__main__":
    input_data_path = "./toy_data/toy_source.json"
    output_data_path = "./toy_data/test_min_neg.json"
    model_path = "BAAI/bge-m3"
    tokenizer_path = "BAAI/bge-m3"
    test_mining = MiningNegativeSamples(
        model_path, tokenizer_path, input_data_path=input_data_path, output_data_path=output_data_path
    )
    test_mining.mining()
