#!/usr/bin/env bash
# -*- coding:utf-8 -*-

export DEVICE=0
export model_path=pd_models
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14lap --model ${model_path}/absa_14lap --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14res --model ${model_path}/absa_14res --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/15res --model ${model_path}/absa_15res --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/16res --model ${model_path}/absa_16res --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14lap --model ${model_path}/absa_14lap_base --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/14res --model ${model_path}/absa_14res_base --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/15res --model ${model_path}/absa_15res_base --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/absa/16res --model ${model_path}/absa_16res_base --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/mrc_ace04 --model ${model_path}/ent_ace04ent --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/mrc_ace05 --model ${model_path}/ent_ace05ent --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/ace05-rel --model ${model_path}/rel_ace05-rel --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/conll04 --model ${model_path}/rel_conll04_large --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/NYT --model ${model_path}/rel_nyt --batch_size 64 --match_mode set
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/scierc --model ${model_path}/rel_scierc_large --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/event/oneie_ace05_en_event --model ${model_path}/evt_ace05evt --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/event/casie --model ${model_path}/evt_casie --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/entity/conll03 --model ${model_path}/ent_conl.04 --batch_size 64
CUDA_VISIBLE_DEVICES=${DEVICE} python inference.py --data data/text2spotasoc/relation/NYT --model ${model_path}/rel_nyt_base --batch_size 64 --match_mode set


grep "test_offset-ent-F1" pd_models/ent*/test_results.txt
grep "test_offset-rel-strict-F1" pd_models/rel*/test_results.txt
grep "test_string-rel-boundary-F1" pd_models/*nyt*/test_results.txt
grep "test_offset-evt-trigger-F1" pd_models/evt*/test_results.txt
grep "test_offset-evt-role-F1" pd_models/evt*/test_results.txt
grep "test_offset-rel-strict-F1" pd_models/absa*/test_results.txt
