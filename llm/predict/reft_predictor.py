# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import argparse
from functools import partial
from types import SimpleNamespace

import paddle
import paddle.incubate.multiprocessing as mp
from paddlenlp.peft.reft import ReFTModel
from paddlenlp.peft.reft import do_predict
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import sys
sys.path.append('/home/ldn/baidu/pyreft/paddle-version/mypr/0705/reft-pr/simple-reft/PaddleNLP/llm')
from utils.data import convert_example_for_reft

device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"


def get_intervention_info(reft_config_file):
    with open(os.path.join(reft_config_file, 'config.json'), "r") as f:
        intervention_info = json.load(f)
    intervention_info['num_interventions'] = len(intervention_info['representations'])
    return intervention_info

def reft_predict(predictor_args):
    intervention_info = get_intervention_info(predictor_args.reft_path)
    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path,
        padding_side="right",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dev_ds = load_dataset(
                "json",
                data_files=os.path.join(predictor_args.dataset_name_or_path, "dev.json"),
            )[0]
    trans_func = partial(convert_example_for_reft, tokenizer=tokenizer, data_args=SimpleNamespace(**{"max_length": predictor_args.max_length, "src_length": predictor_args.src_length}), positions=intervention_info['position'], num_interventions=intervention_info['num_interventions'])
    # dev_ds = dev_ds.map(partial(trans_func,))
    
    
    dev_ds = dev_ds.map(
        partial(
            trans_func,
            is_test=True,
            zero_padding=False,
            flash_mask=False
        )
    )
    
    model = AutoModelForCausalLM.from_pretrained(predictor_args.model_name_or_path, dtype=paddle.bfloat16)
    reft_model = ReFTModel.load(predictor_args.reft_path, model)
    do_predict(
        intervenable=reft_model,
        tokenizer=tokenizer,
        eval_dataset=dev_ds,
        batch_size=predictor_args.batch_size,
        predict_path=predictor_args.output_file,
    )


def get_pred_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='The base model name or path'
    )
    parser.add_argument(
        '--reft_path',
        type=str,
        help='The reft model path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='The output file path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='The batch size in prediction'
    )
    parser.add_argument(
        '--dataset_name_or_path',     
        type=str,
        help='The dataset name or path'
    )
    parser.add_argument(  
        '--max_length',  
        type=int,  
        default=1024,   
        help='The maximum length of input sequences'  
    )  
    parser.add_argument(  
        '--src_length',  
        type=int,  
        default=512,  
        help='The source sequence length'  
    )  
        
    return parser.parse_args()


if __name__ == "__main__":
    predictor_args = get_pred_parser()
    print(predictor_args)
    reft_predict(predictor_args)