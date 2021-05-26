#! /bin/bash

# 1. preprocess dataset by the following. It will produce data/cosql_data_removefrom/

python3 preprocess.py --dataset=cosql --remove_from

# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR

GLOVE_PATH="/home/caiyitao/glove.840B.300d.txt" # you need to change this
LOGDIR="logs_cosql_editsql"

CUDA_VISIBLE_DEVICES=1 python3 run.py --raw_train_filename="data/cosql_data_removefrom/train.pkl" \
          --raw_validation_filename="data/cosql_data_removefrom/dev.pkl" \
          --database_schema_filename="data/cosql_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_cosql_removefrom" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --discourse_level_lstm=1 \
          --use_schema_encoder=1 \
          --use_schema_attention=1 \
          --use_bert=1 \
          --fine_tune_bert=1 \
          --bert_type_abb=uS \
          --interaction_level=1 \
          --reweight_batch=1 \
          --train=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_query_attention=1 \
          --use_previous_query=1 \
          --use_encoder_attention=1 \
          --use_utterance_attention=1 \
          --use_predicted_queries=1

# 3. get evaluation result

python3 postprocess_eval.py --dataset=cosql --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from
