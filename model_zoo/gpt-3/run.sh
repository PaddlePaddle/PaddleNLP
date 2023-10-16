
#!/usr/bin/sh
backend="torch"
seq_len=32768
nnodes=1
var_seq_len="false"
var_nodes="false"



if [[ $# > 0 ]]; then
  backend=$1
fi

if [[ $# > 1 ]]; then
  seq_len=$2
fi
sep_len_list=($seq_len)

if [[ $# > 2 ]]; then
  nnodes=$3
fi
nnodes_list=($nnodes)

if [[ $# > 3 ]]; then
  var_seq_len=$4
  if [[ $var_seq_len == "true" ]]; then
    echo "var_seq_len true"
    sep_len_list=(2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
    # sep_len_list=(2048 4096)
  fi
fi

if [[ $# > 4 ]]; then
  var_nodes=$5
  if [[ $var_nodes == "true" ]]; then
    # nnodes_list=(1 2 4)
    nnodes_list=(1 2)
  fi
fi


echo "backend:${backend}, seq_len:${seq_len}, gpu_num:${gpu_num}, nnodes:${nnodes}, var_seq_len:${var_seq_len}, var_nodes:${var_nodes}"

virtual_env_cmd="source /home/pangengzheng/develop/py39/bin/activate"
for nnodes in ${nnodes_list[@]}; do
  if [ $nnodes > 1 ] && [ -f /root/paddlejob/workspace/hostfile${nnodes} ] && [ -f /root/paddlejob/workspace/hostfile ]; then
    /bin/cp /root/paddle_cloud/workspace/hostfile${nnodes} /root/paddlejob/workspace/hostfile
  fi
  gpu_num=$((8 * $nnodes))
  for seq_len in ${sep_len_list[@]}; do
    if [[ $backend == "torch" ]]; then
      cmd="bash ds_pretrain_gpt_1.3B_seq_parallel_32k.sh ${seq_len} ${gpu_num} ${nnodes}"
      # echo "torch cmd:$cmd"
    else
      gpus="0,1,2,3,4,5,6,7"
      mode="sep"
      # cmd="${virtual_env_cmd}; bash do_run.sh ${mode} ${seq_len} ${gpus} ${gpu_num} ${nnodes}"
      cmd="bash projects/gpt/pretrain_gpt_1.3B_sep2.sh ${seq_len} ${gpu_num} ${nnodes}"
      # echo "paddle cmd:$cmd"
    fi
    if [[ $nnodes > 1 ]]; then
      cmd="mpirun ${cmd}"
    fi
    echo "execute cmd:$cmd"
    eval ${cmd}
  done
done

