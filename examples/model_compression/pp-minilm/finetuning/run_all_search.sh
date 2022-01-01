# $1 means GENERAL_DIR
mkdir -p $1/afqmc
mkdir -p $1/tnews
mkdir -p $1/ifly
mkdir -p $1/ocnli
mkdir -p $1/cmnli
mkdir -p $1/wsc
mkdir -p $1/csl

# The penultimate parameter is the card id, this script can be changed if necessary
bash run_one_search.sh $1 afqmc 0 &
bash run_one_search.sh $1 tnews 1 &
bash run_one_search.sh $1 ifly 2 &
bash run_one_search.sh $1 ocnli 3 &
bash run_one_search.sh $1 csl 4 &
bash run_one_search.sh $1 wsc 5 &

# Because the CMNLI data set is significantly larger than other data sets,
# It needs to be placed on different cards.
lr=1e-4
bs=16
sh run_clue.sh CMNLI $lr $bs 3 128 0 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=32
sh run_clue.sh CMNLI $lr $bs 3 128 1 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=64
sh run_clue.sh CMNLI $lr $bs 3 128 2 $1  > $1/cmnli/${lr}_${bs}_3_128.log &

lr=5e-5
bs=16
sh run_clue.sh CMNLI $lr $bs 3 128 3 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=32
sh run_clue.sh CMNLI $lr $bs 3 128 4 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=64
sh run_clue.sh CMNLI $lr $bs 3 128 5 $1  > $1/cmnli/${lr}_${bs}_3_128.log &

lr=3e-5
bs=16
sh run_clue.sh CMNLI $lr $bs 3 128 6 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=32
sh run_clue.sh CMNLI $lr $bs 3 128 5 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
bs=64
sh run_clue.sh CMNLI $lr $bs 3 128 7 $1  > $1/cmnli/${lr}_${bs}_3_128.log &
