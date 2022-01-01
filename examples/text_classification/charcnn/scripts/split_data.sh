#!/bin/bash

# Function: split the original training set to train and dev.
# Usage (eg.): bash split_data.sh data/ag_news/train.csv

set -x

in=$1
train="$in.split_train"
val="$in.split_val"
awk -v train="$train" -v val="$val" '{if(rand()<0.95) {print > train} else {print > val}}' $in
