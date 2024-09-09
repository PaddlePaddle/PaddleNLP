python preprocess.py --data_dir data/training_data/annotations \
                                    --data_split train \
                                    --output_dir data \
                                    --model_name_or_path bert-base-uncased \
                                    --max_len 510

python preprocess.py --data_dir data/testing_data/annotations \
                                    --data_split test \
                                    --output_dir data \
                                    --model_name_or_path bert-base-uncased \
                                    --max_len 510

cat data/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > data/labels.txt