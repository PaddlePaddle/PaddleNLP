# Whether to use mkl or gpu 
WITH_MKL=ON
WITH_GPU=ON

# Please set:
# * Corresponding PaddlePaddle inference lib
# * Corresponding CUDA lib
# * Corresponding CUDNN lib
# * Corresponding model directory
# * Corresponding vocab directory
# * Corresponding data directory
LIB_DIR=YOUR_LIB_DIR
CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
MODEL_DIR=YOUR_MODEL_DIR
VOCAB_DIR=~/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708
DATA_DIR=~/.paddlenlp/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en

sh run_impl.sh ${LIB_DIR} transformer_e2e ${MODEL_DIR} ${WITH_MKL} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${VOCAB_DIR} ${DATA_DIR}
