#! /bin/bash

# 1. preprocess dataset by the following. It will produce data/sparc_data_removefrom/

#python3 preprocess.py --dataset=sparc --remove_from

# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR

GLOVE_PATH="data/glove.840B.300d.txt" # you need to change this
LOGDIR="logs_sparc_editsql"

CUDA_VISIBLE_DEVICES=1 python3 run.py --raw_train_filename="data/sparc_data_removefrom/train.pkl" \
          --raw_validation_filename="data/sparc_data_removefrom/dev.pkl" \
          --database_schema_filename="data/sparc_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_sparc_removefrom" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --discourse_level_lstm=0 \
          --use_schema_encoder=1 \
          --use_schema_attention=1 \
          --use_encoder_attention=1 \
          --use_bert=1 \
          --bert_type_abb=uS \
          --fine_tune_bert=1 \
          --interaction_level=1 \
          --reweight_batch=1 \
          --freeze=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_predicted_queries=1 \
          --use_previous_query=1 \
          --use_query_attention=1 \
          --use_utterance_attention=1 \
          --save_file="" \
          --reload_embedding=1

# 3. get evaluation result

python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from
