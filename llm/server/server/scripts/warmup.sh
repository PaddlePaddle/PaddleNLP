WARMUP_SEQ_LEN=${1-"8192"}
WARMUP_BATCH_SIZE=${2-"1"}

python warmup.py --batch_size=1 --input_length=${WARMUP_SEQ_LEN}
sleep 1
for ((i=1; i<=WARMUP_BATCH_SIZE; i++)); do
  curl 127.0.0.1:9965/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"text": "hello, llm"}' &
done
sleep 5
python warmup.py --batch_size=1 --input_length=${WARMUP_SEQ_LEN}