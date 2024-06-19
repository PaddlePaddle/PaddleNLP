###
 # This script is used to evaluate the performance of the mrc model (F1)
### 
MODELS=("roberta_base" "roberta_large")
MODES=("attention" "integrated_gradient")

for BASE_MODEL in ${MODELS[*]};
do
    for INTER_MODE in ${MODES[*]};
    do
        for LANGUAGE in "en" "ch";
        do
            echo ${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}
            
            GOLDEN_PATH=../golden/mrc_${LANGUAGE}.tsv
            if [[ $LANGUAGE == "ch" ]]; then
                PRED_FILE=../../rationale_extraction/evaluation_data/mrc/${BASE_MODEL}_${INTER_MODE}_${LANGUAGE}
            else
                PRED_FILE=../../task/mrc/output/mrc_en.${BASE_MODEL}/predict_ans
            fi

            python3 mrc_f1_evaluate.py \
                --golden_path $GOLDEN_PATH \
                --pred_file $PRED_FILE \
                --language $LANGUAGE
        done
    done
done

