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


for task in "ArguAna" "ClimateFEVER" "DBPedia" "FEVER" "FiQA2018" "HotpotQA" "MSMARCO" "NFCorpus" "NQ" "QuoraRetrieval" "SCIDOCS" "SciFact" "Touche2020" "TRECCOVID" "CQADupstackAndroidRetrieval" "CQADupstackEnglishRetrieval" "CQADupstackGamingRetrieval" "CQADupstackGisRetrieval" "CQADupstackMathematicaRetrieval" "CQADupstackPhysicsRetrieval" "CQADupstackProgrammersRetrieval" "CQADupstackStatsRetrieval" "CQADupstackTexRetrieval" "CQADupstackUnixRetrieval" "CQADupstackWebmastersRetrieval" "CQADupstackWordpressRetrieval" "MSMARCOTITLE"
do

       # 1. RocketQA V1
       python3.10 -u eval_mteb.py \
              --corpus_model_name_or_path rocketqa-en-base-v1/passage_model \
              --query_model_name_or_path rocketqa-en-base-v1/query_model \
              --model_flag RocketQA-V1 \
              --output_folder en_results/rocketqa-en-base-v1 \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --query_instruction "" \
              --document_instruction "" \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --dtype "float32" \
              --padding_side right \
              --pooling_method "cls"

       # 2. RocketQA V2     
       python3.10 -u eval_mteb.py \
              --corpus_model_name_or_path rocketqa-en-base-v2/passage_model \
              --query_model_name_or_path rocketqa-en-base-v2/query_model \
              --model_flag RocketQA-V2 \
              --output_folder en_results/rocketqa-en-base-v2 \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --query_instruction "" \
              --document_instruction "" \
              --max_seq_length 512 \
              --eval_batch_size 128 \
              --dtype "float32" \
              --padding_side right \
              --pooling_method "cls"

       # 3. BGE
       python3.10 eval_mteb.py \
              --base_model_name_or_path BAAI/bge-large-en-v1.5 \
              --output_folder en_results/bge-large-en-v1.5 \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --document_instruction 'Represent this sentence for searching relevant passages: ' \
              --pooling_method mean \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --padding_side right \
              --add_bos_token 0 \
              --add_eos_token 0

       # 4. RepLLaMA
       python3.10 eval_mteb.py \
              --base_model_name_or_path castorini/repllama-v1-7b-lora-passage \
              --output_folder en_results/repllama-v1-7b-lora-passage \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --query_instruction 'query: ' \
              --document_instruction 'passage: ' \
              --pooling_method last \
              --max_seq_length 512 \
              --eval_batch_size 2 \
              --padding_side right \
              --add_bos_token 0 \
              --add_eos_token 1

       # 5. NV-Embed-v1
       python3.10 eval_mteb.py \
              --base_model_name_or_path nvidia/NV-Embed-v1 \
              --output_folder en_results/nv-embed-v1 \
              --query_instruction "Given a claim, find documents that refute the claim" \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --eval_batch_size 8

       # 6. BGE-EN-ICL
       python3.10 eval_mteb.py \
              --base_model_name_or_path BAAI/bge-en-icl \
              --output_folder en_results/bge-en-icl \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --query_instruction $'<instruct> Given a scientific claim, retrieve documents that support or refute the claim.\n<query>' \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --dtype "float32" \
              --padding_side left \
              --add_bos_token 1 \
              --add_eos_token 1

       # 7. LLARA-passage
       python3.10 eval_mteb.py \
              --base_model_name_or_path BAAI/LLARA-passage \
              --output_folder en_results/llara-passage \
              --task_name "$task" \
              --task_split $(if [[ "$task" == *"MSMARCO"* ]]; then echo "dev"; else echo "test"; fi) \
              --eval_batch_size 8 \
              --pooling_method last_8 \
              --model_flag llara \
              --add_bos_token 1 \
              --add_eos_token 0 \
              --max_seq_length 532

done