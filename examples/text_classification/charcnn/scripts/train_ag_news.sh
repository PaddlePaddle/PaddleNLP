python train.py \
  --train_path=data/ag_news_csv/train.csv.split_train \
  --val_path=data/ag_news_csv/train.csv.split_val \
  --save_folder=output/models_ag_news \
  --data_augment=True \
  --is_small=True \
#  --geo_aug=True
#  --init_from_ckpt output/models_ag_news/CharCNN_best.pth.tar