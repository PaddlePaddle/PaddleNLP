export TASK_NAME=$1
export LR=$2
export BATCH_SIZE=$3
export PRE_EPOCHS=$4
export SEQ_LEN=$5
export CUDA_VISIBLE_DEVICES=$6
export STUDENT_DIR=$7
export WIDTH_LIST=$8

python -u ./run_ofa.py --model_type ernie \
          --model_name_or_path ${STUDENT_DIR} \
          --task_name $TASK_NAME --max_seq_length ${SEQ_LEN}     \
          --batch_size ${BATCH_SIZE}       \
          --learning_rate ${LR}     \
          --num_train_epochs ${PRE_EPOCHS}     \
          --logging_steps 100     \
          --save_steps 100     \
          --output_dir ./ofa_models/$TASK_NAME/0.75/best_model/ \
          --device gpu  \
          --width_mult_list ${WIDTH_LIST}

