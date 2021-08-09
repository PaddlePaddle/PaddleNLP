
export CUDA_VISIBLE_DEVICES=3

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/ssd2/tianxin04/PaddleNLP/"

test_sets=("public_test_A" "public_test_B")

#strategy_name="rdrop0_bs32_lr2E-5"
#best_model_step="8100"

strategy_name="rdrop0.1_bs32_lr2E-5"
best_model_step="10100"

strategy_name="rdrop0.2_bs32_lr2E-5"
best_model_step="10400"

model_path="./checkpoints_${strategy_name}/model_${best_model_step}/"

for test_set in ${test_sets[@]}; do
	predict_result="${strategy_name}_${best_model_step}.predict_${test_set}"
	echo ${predict_result}
	${PYTHON_BIN} -u \
			predict.py \
			--device gpu \
			--params_path "${model_path}/model_state.pdparams" \
			--batch_size 128 \
			--input_file "./data_v2/test/${test_set}" \
			--result_file ${predict_result}
done
