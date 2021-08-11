
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/ssd2/tianxin04/PaddleNLP/"

test_sets=("public_test_A" "public_test_B")

strategy_name="rdrop0_bs32_lr2E-5"
best_model_step="8100"

strategy_name="rdrop0.1_bs32_lr2E-5"
best_model_step="10100"

strategy_name="rdrop0.2_bs32_lr2E-5"
best_model_step="10400"

#strategy_name="ernie1p0_rdrop0_bs32_lr2E-5"
#best_model_step="11200"

#strategy_name="bert-base-chinese_rdrop0_bs32_lr2E-5"
#best_model_step="9700"

model_path="./checkpoints_${strategy_name}/model_${best_model_step}/"
data_dir="data_v4"

for test_set in ${test_sets[@]}; do
	predict_result="${strategy_name}_${best_model_step}.predict_${test_set}"
	echo ${predict_result}
	${PYTHON_BIN} -u \
			predict.py \
			--device gpu \
			--params_path "${model_path}/model_state.pdparams" \
			--batch_size 128 \
			--input_file "./${data_dir}/test/${test_set}" \
			--result_file ${predict_result}
done
