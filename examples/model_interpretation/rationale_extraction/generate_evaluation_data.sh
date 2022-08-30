###
 # This script concatenates results from previous running to generate a formated result for evaluation use
### 

BASE_MODEL=$1
INTER_MODE=$2
LANGUAGE=$3
TASK=$4

PRED_PATH=../task/${TASK}/output/${TASK}_${LANGUAGE}.${BASE_MODEL}/interpret.${INTER_MODE}
SAVE_PATH=./evaluation_data/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}

SAVE_DIR=./evaluation_data/${TASK}/
[ -d $SAVE_DIR ] || mkdir -p $SAVE_DIR

python3 generate_evaluation_data.py \
    --data_dir ./prediction/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE} \
    --data_dir2 ./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE} \
    --pred_path $PRED_PATH \
    --save_path $SAVE_PATH \
    --inter_mode $INTER_MODE \
    --base_model $BASE_MODEL \
    --language $LANGUAGE