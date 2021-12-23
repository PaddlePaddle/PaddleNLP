OUTPUT_DIR=$1
TASK_NAME=$2

mkdir ${OUTPUT_DIR}/afqmc
mkdir ${OUTPUT_DIR}/tnews
mkdir ${OUTPUT_DIR}/ifly
mkdir ${OUTPUT_DIR}/ocnli
mkdir ${OUTPUT_DIR}/wsc
mkdir ${OUTPUT_DIR}/csl
mkdir ${OUTPUT_DIR}/cmnli


for lr in 1e-4 5e-5 3e-5
do
    for bs in 16 32 64
    do
    echo bs: $bs, lr: $lr

    if [ $TASK_NAME == afqmc ]
    then
    sh run_clue.sh AFQMC $lr $bs 3 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/afqmc/${lr}_${bs}_3_128.log
    fi

    if [ $TASK_NAME == tnews ]
    then
    sh run_clue.sh TNEWS $lr $bs 3 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/tnews/${lr}_${bs}_3_128.log
    fi

    if [ $TASK_NAME == ifly ]
    then
    sh run_clue.sh IFLYTEK $lr $bs 6 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/ifly/${lr}_${bs}_6_128.log
    fi

    if [ $TASK_NAME == ocnli ]
    then
    sh run_clue.sh OCNLI $lr $bs 6 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/ocnli/${lr}_${bs}_6_128.log
    fi

    if [ $TASK_NAME == wsc ]
    then
    sh run_clue.sh CLUEWSC2020 $lr $bs 50 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/wsc/${lr}_${bs}_50_128.log
    fi

    if [ $TASK_NAME == csl ]
    then
    sh run_clue.sh CSL $lr $bs 8 256 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/csl/${lr}_${bs}_8_256.log
    fi

    if [ $TASK_NAME == cmnli ]
    then
    sh run_clue.sh CMNLI $lr $bs 3 128 $3 ${OUTPUT_DIR} > ${OUTPUT_DIR}/cmnli/${lr}_${bs}_3_128.log
    fi
    done
done

