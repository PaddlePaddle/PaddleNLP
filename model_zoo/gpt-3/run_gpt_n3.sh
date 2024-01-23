export NCCL_IB_GID_INDEX=3

export log_dir=log_new
node_1="yq02-inf-sci-k8s-a100-aa2ni5-0071.yq02.baidu.com"
node_2="yq01-inf-hic-k8s-a100-aa24-0009.yq01.baidu.com"
node_3="yq02-inf-sci-k8s-a100-aa2ni5-0065.yq02.baidu.com"

# prepare for ckpt
target_dir="/ywt01_heter/PaddleNLP/model_zoo/gpt-3/output/"

prepare_ckpt() {
    hostname=$(hostname)
    ckpt_dir="output"
    latest_ckpt_dir=""

	latest_ckpt_dir=$(ls -t "$ckpt_dir" | grep -v "^\.$" | head -1)

    if [ "$hostname" = "$node_1" ]; then
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_2:$target_dir/$latest_ckpt_dir >run.log 2>&1
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_3:$target_dir/$latest_ckpt_dir >>run.log 2>&1

        scp -P 8020 -r $node_2:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
        scp -P 8020 -r $node_3:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
    elif [ "$hostname" = "$node_2" ]; then
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_1:$target_dir/$latest_ckpt_dir >>run.log 2>&1
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_3:$target_dir/$latest_ckpt_dir >>run.log 2>&1

        scp -P 8020 -r $node_1:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
        scp -P 8020 -r $node_3:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
    elif [ "$hostname" = "$node_3" ]; then
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_1:$target_dir/$latest_ckpt_dir >>run.log 2>&1
        #scp -P 8020 -r $target_dir/$latest_ckpt_dir $node_2:$target_dir/$latest_ckpt_dir >>run.log 2>&1

        if [ "$(ls -A output/$latest_ckpt_dir)" ]; then
            scp -P 8020 -r $node_1:$target_dir/$latest_ckpt_dir ./ >>run.log 2>&1
            scp -P 8020 -r $node_2:$target_dir/$latest_ckpt_dir ./ >>run.log 2>&1
        else
            scp -P 8020 -r $node_1:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
            scp -P 8020 -r $node_2:$target_dir/$latest_ckpt_dir output/ >>run.log 2>&1
        fi
    fi
    echo $latest_ckpt_dir
}

latest_ckpt=$(prepare_ckpt)

echo "latest_ckpt", $latest_ckpt

#rm -rf $log_dir
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    --auto_cluster_config true \
    --master=10.95.147.146:8091 \
    --nnodes=3 ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_full_auto_parallel_n3.yaml

