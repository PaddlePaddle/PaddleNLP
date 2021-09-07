export CUDA_VISIBLE_DEVICES=0

model_name_or_path=ernie-1.0
checkpoints_path=checkpoints
model_file=best_model.pdparams

# Download SIGHAN test dataset
if [ ! -d "./sighan_test" ]; then
  python download.py
fi

# Predict the test input from sighan13, sighan14, sighan15
for version in 13 14 15
do
python predict_sighan.py --model_name_or_path $model_name_or_path       \
    --test_file sighan_test/sighan$version/input.txt --batch_size 32    \
    --ckpt_path $checkpoints_path/$model_file                           \
    --predict_file predict_sighan$version.txt
done

# Evaluate the prediction result of the model
for version in 13 14 15
do
echo -e "Sighan$version Performace\n"
python sighan_evaluate.py -p predict_sighan$version.txt -t sighan_test/sighan$version/truth.txt
done