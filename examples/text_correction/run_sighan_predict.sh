export CUDA_VISIBLE_DEVICES=3

model_type=ernie
model_name_or_path=ernie-1.0
checkpoints_dir=ernie_log

# model_type=ernie_gram
# model_name_or_path=ernie-gram-zh
# checkpoints_dir=ernie_gram_log

# model_type=roberta
# model_name_or_path=roberta-wwm-ext
# checkpoints_dir=roberta_log

checkpoints_path=$checkpoints_dir/checkpoints5e-5
model_file=best_model.pdparams

# predict the test input from sighan13, sighan14, sighan15
for version in 13 14 15
do
python predict_sighan.py --model_type $model_type --model_name_or_path $model_name_or_path  \
    --test_file sighan_test/sighan$version/input.txt --batch_size 32                    \
    --init_checkpoint_path $checkpoints_path/$model_file --predict_file predict$version.txt
done

# evaluate the prediction of the model
for version in 13 14 15
do
echo -e "Sighan$version Performace\n"
python sighan_evaluate.py -p predict$version.txt -t sighan_test/sighan$version/truth.txt

done