# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from functools import partial

import paddle
from base_model import SemanticIndexBaseStatic
from config import collection_name, embedding_name, partition_tag
from data import convert_example, create_dataloader
from milvus_util import RecallByMilvus

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer


def search_in_milvus(text_embedding):
    recall_client = RecallByMilvus()
    result = recall_client.search(
        text_embedding.numpy(),
        embedding_name,
        collection_name,
        partition_names=[partition_tag],
        output_fields=["pk", "text"],
    )
    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, text field: {hit.entity.get('text')}")


if __name__ == "__main__":
    device = "gpu"
    max_seq_length = 64
    output_emb_size = 256
    batch_size = 1
    params_path = "checkpoints/model_40/model_state.pdparams"
    id2corpus = {0: "国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据"}
    model_name_or_path = "rocketqa-zh-base-query-encoder"
    paddle.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]
    pretrained_model = AutoModel.from_pretrained(model_name_or_path)
    model = SemanticIndexBaseStatic(pretrained_model, output_emb_size=output_emb_size)
    # Load pretrained semantic model
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")
    # convert_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)
    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    # Need better way to get inner model of DataParallel
    all_embeddings = []
    model.eval()
    with paddle.no_grad():
        for batch_data in corpus_data_loader:
            input_ids, token_type_ids = batch_data
            text_embeddings = model.get_pooled_embedding(input_ids, token_type_ids)
            all_embeddings.append(text_embeddings)
    text_embedding = all_embeddings[0]
    print(text_embedding.shape)
    print(text_embedding)
    search_in_milvus(text_embedding)
