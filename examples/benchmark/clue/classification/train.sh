# python train.py --task_name cluewsc2020 \
#                 --learning_rate 1e-5 \
#                 --num_train_epochs 50

# CUDA_VISIBLE_DEVICES=2 python train.py --task_name ocnli \
#                 --learning_rate 2e-5 \
#                 --num_train_epochs 3


# CUDA_VISIBLE_DEVICES=2 python train.py --task_name csl \
#                 --learning_rate 2e-5 \
#                 --num_train_epochs 3

# CUDA_VISIBLE_DEVICES=2 python train.py --task_name cmnli \
#                 --learning_rate 2e-5 \
#                 --num_train_epochs 3

# CUDA_VISIBLE_DEVICES=2 python train.py --task_name iflytek \
#                 --learning_rate 2e-5 \
#                 --num_train_epochs 3

# CUDA_VISIBLE_DEVICES=2 python train.py --task_name tnews \
#                 --learning_rate 2e-5 \
#                 --num_train_epochs 3

CUDA_VISIBLE_DEVICES=2 python train.py --task_name afqmc \
                --learning_rate 2e-5 \
                --num_train_epochs 3
