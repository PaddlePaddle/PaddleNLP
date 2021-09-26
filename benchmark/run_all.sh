# Test training benchmark for several models.

# Use docker： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37

# Usage:
#   git clone git clone https://github.com/PaddlePaddle/PaddleSeg.git
#   cd PaddleSeg
#   bash benchmark/run_all.sh

pip install -r requirements.txt
pip install regex sentencepiece tqdm visualdl
pip install -e ./

# Download test dataset and save it to PaddleSeg/data
# It automatic downloads the pretrained models saved in ~/.paddleseg

mkdir -p data && cd data
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
cd -

model_name_list=(gpt2-en)
fp_item_list=(fp16 fp32)     # set fp32 or fp16, segformer_b0 doesn't support fp16 with Paddle2.1.2
bs_list=(8) # FP16 could use bs=16 for 32G v100
max_iters=200 # control the test time


for model_name in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name}
            sleep 60

            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} \
                ${max_iters} ${model_name} 
            sleep 60
            done
      done
done

rm -rf data/*
