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

import numpy as np
import paddle
import paddle.distributed as dist
from tqdm import tqdm

from paddlenlp.transformers import AutoConfig, AutoModel, AutoTokenizer


class Embedding_Evaluation:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        query_pos_passage_path,
        neg_passage_path,
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

        self.query_pos_passage_path = query_pos_passage_path
        self.neg_passage_path = neg_passage_path
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

    def evaluate(self):
        query_data_list = []
        pos_passage_data_list = []
        with open(self.query_pos_passage_path, "r") as f:
            for line in f:
                single_data = json.loads(line)
                query_data_list.append(single_data["query"])
                pos_passage_data_list.append(single_data["pos_passage"][0])

        neg_passage_data_list = []
        with open(self.neg_passage_path, "r") as f:
            for line in f:
                single_data = json.loads(line)
                neg_passage_data_list.append(single_data["neg_passage"][0])

        passage_data_list = pos_passage_data_list + neg_passage_data_list

        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        query_chunk_size = len(query_data_list) // world_size
        passage_chunk_size = len(passage_data_list) // world_size
        if rank == world_size - 1:
            # The last process handles the remaining data
            query_data_chunk = query_data_list[rank * query_chunk_size :]
            passage_data_chunk = passage_data_list[rank * passage_chunk_size :]
        else:
            query_data_chunk = query_data_list[rank * query_chunk_size : (rank + 1) * query_chunk_size]
            passage_data_chunk = passage_data_list[rank * passage_chunk_size : (rank + 1) * passage_chunk_size]

        batch_size = 4  # Adjust batch size according to your hardware and needs
        local_p_vecs = []
        local_q_vecs = []

        # Use tqdm to iterate over query_data_chunk and get embeddings in batches
        for batch in tqdm(range(0, len(passage_data_chunk), batch_size), desc="Processing passage embeddings"):
            batch_start = batch
            batch_end = min(batch_start + batch_size, len(passage_data_chunk))
            batch_texts = passage_data_chunk[batch_start:batch_end]

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

            query_embedding_tensor = paddle.to_tensor(q_vecs, dtype=self.dtype)
            passage_embedding_tensor = paddle.to_tensor(p_vecs, dtype=self.dtype)
            similarity_matrix = self.calculate_cosine_similarity_matrix(
                query_embedding_tensor, passage_embedding_tensor
            )
            query_num = len(query_data_list)
            true_answers = [i for i in range(query_num)]
            hit_count_10, hit_coun_5, hit_count_3, hit_count_1 = 0, 0, 0, 0
            reciprocal_rank_sum_10, reciprocal_rank_sum_5, reciprocal_rank_sum_3 = 0, 0, 0
            ndcg_10, ndcg_5, ndcg_3 = 0.0, 0.0, 0.0
            for i in range(query_num):
                similarities = similarity_matrix[i]

                # get the sorted indices
                sorted_indices = paddle.argsort(-similarities)

                # find the index of the true answer
                true_answer_index = true_answers[i]
                rank = paddle.where(sorted_indices == true_answer_index)[0][0] + 1  # rank starts from 1

                if rank <= 10:
                    hit_count_10 += 1
                    reciprocal_rank_sum_10 += 1.0 / rank
                if rank <= 5:
                    hit_coun_5 += 1
                    reciprocal_rank_sum_5 += 1.0 / rank
                if rank <= 3:
                    hit_count_3 += 1
                    reciprocal_rank_sum_3 += 1.0 / rank
                if rank <= 1:
                    hit_count_1 += 1

                relevance_scores = [0] * 10
                if rank <= 10:
                    relevance_scores[rank - 1] = 1
                ndcg_10 += self.calculate_ndcg(relevance_scores[:10], k=10)
                ndcg_5 += self.calculate_ndcg(relevance_scores[:5], k=5)
                ndcg_3 += self.calculate_ndcg(relevance_scores[:3], k=3)

            print(f"Hit rate when recall Top 10: ({hit_count_10*100./query_num:.2f}%)\n")
            print(f"Hit rate when recall Top 5: ({hit_coun_5*100./query_num:.2f}%)\n")
            print(f"Hit rate when recall Top 3: ({hit_count_3*100./query_num:.2f}%)\n")
            print(f"Hit rate when recall Top 1: ({hit_count_1*100./query_num:.2f}%)\n")
            print(f"MRR when recall Top 10: ({reciprocal_rank_sum_10.item() / query_num:.4f})\n")
            print(f"MRR when recall Top 5: ({reciprocal_rank_sum_5.item() / query_num:.4f})\n")
            print(f"MRR when recall Top 3: ({reciprocal_rank_sum_3.item() / query_num:.4f})\n")
            print(f"NDCG@10: ({ndcg_10/ query_num:.4f})\n")
            print(f"NDCG@5: ({ndcg_5/ query_num:.4f})\n")
            print(f"NDCG@3: ({ndcg_3/ query_num:.4f})\n")

            eval_result_dict = {
                "hit_rate@10": hit_count_10 / query_num,
                "hit_rate@5": hit_coun_5 / query_num,
                "hit_rate@3": hit_count_3 / query_num,
                "hit_rate@1": hit_count_1 / query_num,
                "mrr@10": reciprocal_rank_sum_10.item() / query_num,
                "mrr@5": reciprocal_rank_sum_5.item() / query_num,
                "mrr@3": reciprocal_rank_sum_3.item() / query_num,
                "ndcg@10": ndcg_10 / query_num,
                "ndcg@5": ndcg_5 / query_num,
                "ndcg@3": ndcg_3 / query_num,
            }
            return eval_result_dict

    def calculate_cosine_similarity_matrix(self, query_matrix, answer_matrix):
        """Calculate the cosine similarity between two matrices by processing query vectors one by one."""
        num_queries = query_matrix.shape[0]
        num_answers = answer_matrix.shape[0]

        # Precompute the norms of answer vectors to save computation
        answer_norms = paddle.linalg.norm(answer_matrix, axis=1, keepdim=True)

        # Initialize the similarity matrix with zeros
        similarity_matrix = paddle.zeros((num_queries, num_answers))

        # Process each query vector one by one
        for i in tqdm(range(num_queries)):
            query_vector = query_matrix[i : i + 1]  # Extract the i-th query vector

            # Calculate the norm of the query vector
            query_norm = paddle.linalg.norm(query_vector, axis=1, keepdim=True)

            # Calculate the dot product between the query vector and all answer vectors
            dot_product = paddle.matmul(query_vector, answer_matrix, transpose_y=True)

            # Calculate the cosine similarity for the i-th query vector
            similarity_vector = dot_product / (query_norm * answer_norms.transpose((1, 0)))

            # Update the similarity matrix with the computed similarity vector
            similarity_matrix[i] = similarity_vector

        return similarity_matrix

    def calculate_ndcg(self, relevance_scores, k):
        """Calculate NDCG@k for a given set of relevance scores"""
        # Calculate DCG
        dcg = sum((rel) / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))

        # Calculate IDCG (Ideal DCG)
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        idcg = sum((rel) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance_scores[:k]))

        # Avoid division by zero
        if idcg == 0:
            return 0.0

        # Calculate NDCG
        ndcg = dcg / idcg
        return ndcg


if __name__ == "__main__":
    model_path = "BAAI/bge-m3"
    tokenizer_path = "BAAI/bge-m3"
    query_pos_passage_path = "./toy_data/toy_dev.json"
    neg_passage_path = "./toy_data/toy_dev_neg.json"
    eval = Embedding_Evaluation(model_path, tokenizer_path, query_pos_passage_path, neg_passage_path)
    print(eval.evaluate())
