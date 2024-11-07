# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

TASK=similarity

if [[ $TASK == "mrc" ]]; then
    MODELS=("roberta_base" "roberta_large")
    MODES=("attention" "integrated_gradient")
else
    MODELS=("roberta_large" "roberta_base" "lstm")
    MODES=("lime" "attention" "integrated_gradient")
fi

for BASE_MODEL in ${MODELS[*]};
do
    for INTER_MODE in ${MODES[*]};
    do
        for LANGUAGE in "ch" "en";
        do
            if [[ $LANGUAGE == "ch" ]]; then
                if [[ $TASK == "senti" ]]; then
                    RATIO_DIC="[0.311]"
                elif [[ $TASK == "similarity" ]]; then
                    RATIO_DIC="[0.701,0.709]"
                elif [[ $TASK == "mrc" ]]; then
                    RATIO_DIC="[0.096]"
                fi
            elif [[ $LANGUAGE == "en" ]]; then
                if [[ $TASK == "senti" ]]; then
                    RATIO_DIC="[0.192]"
                elif [[ $TASK == "similarity" ]]; then
                    RATIO_DIC="[0.511,0.505]"
                elif [[ $TASK == "mrc" ]]; then
                    RATIO_DIC="[0.102]"
                fi
            fi
            echo ${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}

            PRED_PATH=../task/${TASK}/output/${TASK}_${LANGUAGE}.${BASE_MODEL}/interpret.${INTER_MODE}
            SAVE_PATH=./rationale/${TASK}/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}
            [ -d $SAVE_PATH ] || mkdir -p $SAVE_PATH

            python3 ./newp_text_generate.py \
                --pred_path $PRED_PATH \
                --save_path $SAVE_PATH \
                --task $TASK \
                --language $LANGUAGE \
                --ratio $RATIO_DIC
            wait

            sh ./run_2_pred_${TASK}_per.sh $BASE_MODEL $INTER_MODE $LANGUAGE
            wait
            
            sh ./generate_evaluation_data.sh $BASE_MODEL $INTER_MODE $LANGUAGE $TASK
            wait
            
            echo ${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}_finished
        done
    done
done
