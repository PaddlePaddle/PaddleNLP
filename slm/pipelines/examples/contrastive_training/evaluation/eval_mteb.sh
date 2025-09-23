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

#!/bin/bash

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
declare -A query_instructions=(
    ["ArguAna"]="Given a claim, find documents that refute the claim."
    ["ClimateFEVER"]="Given a claim about climate change, retrieve documents that support or refute the claim."
    ["DBPedia"]="Given a query, retrieve relevant entity descriptions from DBPedia."
    ["FEVER"]="Given a claim, retrieve documents that support or refute the claim."
    ["FiQA2018"]="Given a financial question, retrieve user replies that best answer the question."
    ["HotpotQA"]="Given a multi-hop question, retrieve documents that can help answer the question."
    ["MSMARCO"]="Given a web search query, retrieve relevant passages that answer the query."
    ["NFCorpus"]="Given a question, retrieve relevant documents that best answer the question."
    ["Natural Question"]="Given a question, retrieve Wikipedia passages that answer the question."
    ["QuoraRetrieval"]="Given a question, retrieve questions that are semantically equivalent to the given question."
    ["SCIDOCS"]="Given a scientific paper title, retrieve paper abstracts that are cited by the given paper."
    ["SciFact"]="Given a scientific claim, retrieve documents that support or refute the claim."
    ["Touche2020"]="Given a question, retrieve detailed and persuasive arguments that answer the question."
    ["TREC-COVID"]="Given a query, retrieve documents that answer the query."
    ["MSMARCOTITLE"]="Given a web search query, retrieve relevant passages that answer the query."
    ["CQADupstackAndroidRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackEnglishRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackGamingRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackGisRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackMathematicaRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackPhysicsRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackProgrammersRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackStatsRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackTexRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackUnixRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackWebmastersRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
    ["CQADupstackWordpressRetrieval"]="Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question."
)

# Define the list of all tasks (datasets) to be evaluated.
TASKS=(
    "ArguAna" "ClimateFEVER" "DBPedia" "FEVER" "FiQA2018" "HotpotQA" "MSMARCO" "NFCorpus" "NQ" "QuoraRetrieval" 
    "SCIDOCS" "SciFact" "Touche2020" "TRECCOVID" "CQADupstackAndroidRetrieval" "CQADupstackEnglishRetrieval" 
    "CQADupstackGamingRetrieval" "CQADupstackGisRetrieval" "CQADupstackMathematicaRetrieval" "CQADupstackPhysicsRetrieval" 
    "CQADupstackProgrammersRetrieval" "CQADupstackStatsRetrieval" "CQADupstackTexRetrieval" "CQADupstackUnixRetrieval" 
    "CQADupstackWebmastersRetrieval" "CQADupstackWordpressRetrieval" "MSMARCOTITLE")

# You can uncomment the models you wish to evaluate.
MODELS_TO_RUN=("RocketQA-V1" "RocketQA-V2" "BGE" "RepLLaMA" "NV-Embed-v1" "BGE-EN-ICL" "LLARA-passage" "Qwen3-Embedding")


# ===================================================================================
# 🚀 1. RocketQA V1
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " RocketQA-V1 " ]]; then
    echo "===== Running Evaluation for Model: RocketQA V1 ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        python3.10 -u evaluation/eval_mteb.py \
              --corpus_model_name_or_path rocketqa-v1-marco-para-encoder \
              --query_model_name_or_path rocketqa-v1-marco-query-encoder \
              --model_flag RocketQA-V1 \
              --output_folder en_results/rocketqa-en-base-v1 \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction "" \
              --document_instruction "" \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --dtype "float32" \
              --padding_side right \
              --pooling_method "cls"
    done
fi


# ===================================================================================
# 🚀 2. RocketQA V2
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " RocketQA-V2 " ]]; then
    echo "===== Running Evaluation for Model: RocketQA V2 ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        python3.10 -u evaluation/eval_mteb.py \
              --corpus_model_name_or_path rocketqav2-en-marco-para-encoder \
              --query_model_name_or_path rocketqav2-en-marco-query-encoder \
              --model_flag RocketQA-V2 \
              --output_folder en_results/rocketqa-en-base-v2 \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction "" \
              --document_instruction "" \
              --max_seq_length 512 \
              --eval_batch_size 128 \
              --dtype "float32" \
              --padding_side right \
              --pooling_method "cls"
    done
fi


# ===================================================================================
# 🎯 3. BGE (BAAI/bge-large-en-v1.5)
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " BGE " ]]; then
    echo "===== Running Evaluation for Model: BGE (bge-large-en-v1.5) ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path BAAI/bge-large-en-v1.5 \
              --output_folder en_results/bge-large-en-v1.5_2 \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction 'Represent this sentence for searching relevant passages: ' \
              --document_instruction "" \
              --pooling_method mean \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --padding_side right \
              --add_bos_token 0 \
              --add_eos_token 0 
    done
fi


# ===================================================================================
# 🦙 4. RepLLaMA
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " RepLLaMA " ]]; then
    echo "===== Running Evaluation for Model: RepLLaMA ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path castorini/repllama-v1-7b-lora-passage \
              --output_folder en_results/repllama-v1-7b-lora-passage \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction 'query: ' \
              --document_instruction 'passage: ' \
              --pooling_method last \
              --max_seq_length 512 \
              --eval_batch_size 2 \
              --padding_side right \
              --add_bos_token 0 \
              --add_eos_token 1
    done
fi


# ===================================================================================
#  Nvidia 5. NV-Embed-v1
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " NV-Embed-v1 " ]]; then
    echo "===== Running Evaluation for Model: NV-Embed-v1 ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        query_instruction="${query_instructions[$task]}"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path nvidia/NV-Embed-v1 \
              --output_folder en_results/nv-embed-v1 \
              --query_instruction "$query_instruction" \
              --document_instruction "" \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --eval_batch_size 8
    done
fi


# ===================================================================================
# 🎯 6. BGE-EN-ICL
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " BGE-EN-ICL " ]]; then
    echo "===== Running Evaluation for Model: BGE-EN-ICL ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        query_instruction="${query_instructions[$task]}"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path BAAI/bge-en-icl \
              --output_folder en_results/bge-en-icl \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction "<instruct> ${query_instruction}\n<query>" \
              --document_instruction "" \
              --max_seq_length 512 \
              --eval_batch_size 32 \
              --dtype "float32" \
              --padding_side left \
              --add_bos_token 1 \
              --add_eos_token 1
    done
fi


# ===================================================================================
# 🦙 7. LLARA-passage
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " LLARA-passage " ]]; then
    echo "===== Running Evaluation for Model: LLARA-passage ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path BAAI/LLARA-passage \
              --output_folder en_results/llara-passage \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction "" \
              --document_instruction "" \
              --eval_batch_size 8 \
              --pooling_method last_8 \
              --model_flag llara \
              --add_bos_token 1 \
              --add_eos_token 0 \
              --max_seq_length 532
    done
fi


# ===================================================================================
# 👀 8. Qwen3-Embedding
# ===================================================================================
if [[ " ${MODELS_TO_RUN[*]} " =~ " Qwen3-Embedding " ]]; then
    echo "===== Running Evaluation for Model: Qwen3-Embedding ====="
    for task in "${TASKS[@]}"; do
        echo "--- Task: $task ---"
        query_instruction="${query_instructions[$task]}"
        python3.10 evaluation/eval_mteb.py \
              --base_model_name_or_path Qwen/Qwen3-Embedding-8B \
              --output_folder en_results/qwen3-embedding \
              --task_name "$task" \
              --task_split $([[ "$task" == *"MSMARCO"* ]] && echo "dev" || echo "test") \
              --query_instruction "Instruct: ${query_instruction}\nQuery:" \
              --document_instruction "" \
              --eval_batch_size 8 \
              --pooling_method last \
              --model_flag qwen3 \
              --add_bos_token 0 \
              --add_eos_token 1 \
              --max_seq_length 4096 
    done
fi


echo "All specified evaluations are complete."