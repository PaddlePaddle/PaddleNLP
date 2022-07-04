#!/bin/bash
function check_iplist() {
    if [ ${iplist:-} ]; then
        export PADDLE_PSERVER_PORT=9184
        export PADDLE_TRAINER_IPS=${iplist} 
        export PADDLE_CURRENT_IP=`hostname -i`
        iparray=(${iplist//,/ })
        for i in "${!iparray[@]}"; do
        if [ ${iparray[$i]} == ${PADDLE_CURRENT_IP} ]; then
            export PADDLE_TRAINER_ID=$i
        fi
        done
        export TRAINING_ROLE=TRAINER
        export PADDLE_INIT_TRAINER_COUNT=${#iparray[@]}
        export PADDLE_PORT=${PADDLE_PSERVER_PORT}
        export PADDLE_TRAINERS=${PADDLE_TRAINER_IPS}
        export POD_IP=${PADDLE_CURRENT_IP}
        export PADDLE_TRAINERS_NUM=${PADDLE_INIT_TRAINER_COUNT}
        export PADDLE_IS_LOCAL=0
        export GLOG_v=0
        export GLOG_logtostderr=1
        export NCCL_DEBUG=INFO
        export NCCL_IB_GID_INDEX=3
    fi
}
