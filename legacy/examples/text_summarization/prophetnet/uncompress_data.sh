tar -xvf ./glge_public.tar
tar -zxvf ./glge_hidden_v1.1.tar.gz

DATA=./data
DATASETS=(cnndm gigaword)
mkdir $DATA
for DATASET in ${DATASETS[@]}; do
  echo $DATASET
mkdir $DATA/$DATASET\_data
mv ./glge-released-dataset/easy/$DATASET\_data/org_data/* $DATA/$DATASET\_data/
mv ./glge-hidden-dataset/easy/$DATASET\_data/org_data/* $DATA/$DATASET\_data/
done
