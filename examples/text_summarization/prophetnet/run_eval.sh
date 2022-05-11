DATASET=$1

if [ $DATASET = cnndm ]
then
python generate.py \
    --dataset=cnndm \
    --model_name_or_path=prophetnet-large-uncased \
    --output_path=./generate/cnndm/generate.txt \
    --min_target_length=45 \
    --max_target_length=110 \
    --decode_strategy=beam_search \
    --num_beams=4 \
    --length_penalty=1.2 \
    --batch_size=16 \
    --ignore_pad_token_for_loss=True \
    --early_stopping=True \
    --logging_steps=100 \
    --device=gpu
else
python generate.py \
    --dataset=gigaword \
    --model_name_or_path=prophetnet-large-uncased \
    --output_path=./generate/gigaword/generate.txt \
    --min_target_length=1 \
    --max_target_length=200 \
    --decode_strategy=beam_search \
    --num_beams=4 \
    --length_penalty=1.6 \
    --batch_size=16 \
    --ignore_pad_token_for_loss=True \
    --early_stopping=True \
    --logging_steps=100 \
    --device=gpu
fi


python eval.py --dataset $DATASET --generated ./generate/$DATASET/generate.txt