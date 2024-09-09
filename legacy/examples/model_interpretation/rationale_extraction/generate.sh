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
