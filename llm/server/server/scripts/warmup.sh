WARMUP_SEQ_LEN=8192


python warmup.py --batch_size=1 --input_length=8192
sleep 1
for i in {1..40}
do
  curl 127.0.0.1:9965/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"text": "hello, llm"}' &
done
sleep 5
python warmup.py --batch_size=1 --input_length=8192