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

unset CUDA_VISIBLE_DEVICES
mkdir -p output
CHECKPOINT_PATH=model_17370/
TEST_SET="../dureader-retrieval-baseline-dataset/dev/dev.q.format" 
DATA_PATH="../dureader-retrieval-baseline-dataset/passage-collection"
CUDA_VISIBLE_DEVICES=0 python inference_de.py \
                    --text_file $TEST_SET \
                    --output_path output \
                    --params_path checkpoint/${CHECKPOINT_PATH}/model_state.pdparams \
                    --output_emb_size 0 \
                    --mode query \
                    --batch_size 256 \
                    --max_seq_length 32
# extract para
for part in 0 1 2 3;do
    TASK_DATA_PATH=${DATA_PATH}/part-0${part}
    count=$((part))
    CUDA_VISIBLE_DEVICES=${count} nohup python inference_de.py \
                        --text_file $TASK_DATA_PATH \
                        --output_path output \
                        --mode title \
                        --params_path checkpoint/${CHECKPOINT_PATH}/model_state.pdparams \
                        --output_emb_size 0 \
                        --batch_size 256 \
                        --max_seq_length 384 >> output/test.log &
    pid[$part]=$!
    echo $part start: pid=$! >> output/test.log
done
wait

for index in 0 1 2 3;do
    python create_index.py --embedding_path output/part-0${index}.npy \
                          --save_path output \
                          --output_index_path para.index.part${index}
done

TOP_K=50
TEST_SET="../dureader-retrieval-baseline-dataset/dev/dev.q.format"
for part in 0 1 2 3;do
    CUDA_VISIBLE_DEVICES=1 python index_search.py --para_index_file output/para.index.part${index} \
                                                  --query_embedding_file output/dev.q.format.npy \
                                                  --query_text_file ${TEST_SET} \
                                                  --topk ${TOP_K} \
                                                  --outfile output/res${TOP_K}.top-part${part}


done
