dataset=$1
device=$2

MAX_LEN=128
dataname=$dataset

case $dataset in
    CoLA)
        temp="{'text':'text_a'} This is {'mask'}."
        verb="{'0':'incorrect','1':'correct'}"
        ;;
    MRPC)
        temp="{'text':'text_a'}{'mask'},{'text':'text_b'}"
        verb="{'0':'No','1':'Yes'}"
        ;;
    QQP)
        temp="{'text':'text_a'}{'mask'},{'text':'text_b'}"
        verb="{'0':'No','1':'Yes'}"
        ;;
    STS-B)
        temp="{'text':'text_a'}{'mask'},{'text':'text_b'}"
        verb="{'0':'No','1':'Yes'}"
        ;;
    MNLI)
        temp="{'text':'text_a'}?{'mask'},{'text':'text_b'}"
        verb="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LEN=256
        ;;
    MNLI-mm)
        temp="{'text':'text_a'}?{'mask'},{'text':'text_b'}"
        verb="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LEN=256
        dataname='MNLI'
        ;;
    SNLI)
        temp="{'text':'text_a'}?{'mask'},{'text':'text_b'}"
        verb="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        MAX_LEN=256
        ;;
    QNLI)
        temp="{'text':'text_a'}?{'mask'},{'text':'text_b'}"
        verb="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    RTE)
        temp="{'text':'text_a'}?{'mask'},{'text':'text_b'}"
        verb="{'not_entailment':'No','entailment':'Yes'}"
        MAX_LEN=256
        ;;
    mr)
        temp="{'text':'text_a'} It was {'mask'}"
        verb="{0:'terrible',1:'great'}"
        MAX_LEN=160
        ;;
    sst-5)
        temp="{'text':'text_a'} It was {'mask'}." 
        temp="{'text':'text_a'} {'mask'}" 
        verb="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        ;;
    SST-2)
        temp="{'text':'text_a'} It was {'mask'}."
        verb="{'0':'terrible','1':'great'}"
        ;;
    subj)
        temp="{'text':'text_a'} This is {'mask'}."
        verb="{0:'subjective',1:'objective'}"
        MAX_LEN=256
        ;;
    trec)
        temp="{'mask'}:{'text':'text_a'}"
        verb="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        ;;
    cr)
        temp="{'text':'text_a'} It was {'mask'}."
        verb="{0:'terrible',1:'great'}"
        MAX_LEN=160
        ;;
    mpqa)
        temp="{'text':'text_a'} It was {'mask'}"
        verb="{0:'terrible',1:'great'}"
        MAX_LEN=128
        ;;

esac

echo $temp
echo $verb


ALPHA=0
for seed in 13 21 42 87 100
do
    for lr in 1e-5 2e-5 5e-5
    do
        for bs in 2 4 8
        do
            CUDA_VISIBLE_DEVICES=$device python rgl.py \
            --output_dir ./ckpt_pet_roberta_$seed/ \
            --dataset $dataset \
            --data_path ./data/k-shot/$dataname/16-$seed/ \
            --max_seq_length $MAX_LEN \
            --max_steps 1000 \
            --logging_step 10 \
            --eval_step 100 \
            --batch_size $bs \
            --alpha $ALPHA \
            --seed $seed \
            --learning_rate $lr \
            --template "$temp" \
            --verbalizer "$verb" \
            --overwrite_output 
        done
    done
done

