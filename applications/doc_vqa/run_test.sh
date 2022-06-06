export CUDA_VISIBLE_DEVICES=0

QUESTION=$1

# Question: NFC咋开门

if [ $# != 1 ];then
    echo "USAGE: sh script/run_cross_encoder_test.sh \$QUESTION"
    exit 1
fi

# compute scores for QUESTION and OCR parsing results  with Rerank module
cd Rerank
bash run_test.sh ${QUESTION}
cd ..

# extraction answer for QUESTION from the top1 of rank
cd Extraction
bash run_test.sh ${QUESTION}
cd ..
