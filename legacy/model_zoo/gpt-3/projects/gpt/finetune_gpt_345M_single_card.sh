# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export CUDA_VISIBLE_DEVICES=0

# Single-Sentence Tasks
if [ $1 == "CoLA" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=CoLA \
      -o Data.Train.dataset.root=./dataset/cola_public/ \
      -o Data.Eval.dataset.name=CoLA \
      -o Data.Eval.dataset.root=./dataset/cola_public/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.metric.train.name=Mcc \
      -o Model.metric.eval.name=Mcc \
      -o Model.num_classes=2
elif [ $1 == "SST2" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=SST2 \
      -o Data.Train.dataset.root=./dataset/SST-2/ \
      -o Data.Eval.dataset.name=SST2 \
      -o Data.Eval.dataset.root=./dataset/SST-2/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
# Similarity and Paraphrase Tasks
elif [ $1 == "MRPC" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Engine.num_train_epochs=5 \
      -o Data.Train.dataset.name=MRPC \
      -o Data.Train.dataset.root=./dataset/MRPC/ \
      -o Data.Eval.dataset.name=MRPC \
      -o Data.Eval.dataset.root=./dataset/MRPC/ \
      -o Data.Eval.dataset.split=test \
      -o Model.num_classes=2 \
      -o Model.metric.train.name=AccuracyAndF1 \
      -o Model.metric.eval.name=AccuracyAndF1
elif [ $1 == "QQP" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=QQP \
      -o Data.Train.dataset.root=./dataset/QQP/ \
      -o Data.Eval.dataset.name=QQP \
      -o Data.Eval.dataset.root=./dataset/QQP/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2 \
      -o Model.metric.train.name=AccuracyAndF1 \
      -o Model.metric.eval.name=AccuracyAndF1
elif [ $1 == "STSB" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=STSB \
      -o Data.Train.dataset.root=./dataset/STS-B/ \
      -o Data.Eval.dataset.name=STSB \
      -o Data.Eval.dataset.root=./dataset/STS-B/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=1 \
      -o Model.metric.train.name=PearsonAndSpearman \
      -o Model.metric.eval.name=PearsonAndSpearman \
      -o Model.loss.train.name=MSELoss \
      -o Model.loss.eval.name=MSELoss
# Inference Tasks
elif [ $1 == "MNLI" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=MNLI \
      -o Data.Train.dataset.root=./dataset/multinli_1.0 \
      -o Data.Eval.dataset.name=MNLI \
      -o Data.Eval.dataset.root=./dataset/multinli_1.0 \
      -o Data.Eval.dataset.split=${2:-"dev_matched"} \
      -o Model.num_classes=3
elif [ $1 == "QNLI" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=QNLI \
      -o Data.Train.dataset.root=./dataset/QNLI/ \
      -o Data.Eval.dataset.name=QNLI \
      -o Data.Eval.dataset.root=./dataset/QNLI/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
elif [ $1 == "RTE" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=RTE \
      -o Data.Train.dataset.root=./dataset/RTE/ \
      -o Data.Eval.dataset.name=RTE \
      -o Data.Eval.dataset.root=./dataset/RTE/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
elif [ $1 == "WNLI" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Engine.num_train_epochs=5 \
      -o Data.Train.dataset.name=WNLI \
      -o Data.Train.dataset.root=./dataset/WNLI/ \
      -o Data.Eval.dataset.name=WNLI \
      -o Data.Eval.dataset.root=./dataset/WNLI/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
else
   echo "Task name not recognized, please input CoLA, SST2, MRPC, QQP, STSB, MNLI, QNLI, RTE, WNLI."
fi
