export CUDA_VISIBLE_DEVICES=3

model_name_or_path=ernie-1.0
checkpoints_path=checkpoints5e-5
model_file=best_model.pdparams

# 下载sighan测试集
if [ ! -d "./sighan_test" ]; then
  python download.py
fi

# predict the test input from sighan13, sighan14, sighan15
for version in 13 14 15
do
python predict_sighan.py --model_name_or_path $model_name_or_path       \
    --test_file sighan_test/sighan$version/input.txt --batch_size 32    \
    --init_checkpoint_path $checkpoints_path/$model_file                \
    --predict_file predict$version.txt
done

# evaluate the prediction of the model
for version in 13 14 15
do
echo -e "Sighan$version Performace\n"
python sighan_evaluate.py -p predict$version.txt -t sighan_test/sighan$version/truth.txt

done