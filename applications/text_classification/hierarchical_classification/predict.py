# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse

import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import get_wos_label_list

parser = argparse.ArgumentParser()
parser.add_argument("--params_path",
                    default="./checkpoint/model_state.pdparams",
                    type=str,
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length "
                    "after tokenization. Sequences longer than this"
                    "will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size",
                    default=12,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--model_name',
                    default='ernie-2.0-base-en',
                    help="Define which model to train, "
                    "defaults to ernie-2.0-base-en.")
args = parser.parse_args()


@paddle.no_grad()
def predict(data, label_list):
    """
    Predicts the data labels.
    Args:

        data (obj:`List`): The processed data whose each element is one sequence.
        label_map(obj:`List`): The label id (key) to label str (value) map.
 
    """
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_classes=len(label_list))
    model.set_dict(paddle.load(os.path.join(args.params_path)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    examples = []
    for text in data:
        result = tokenizer(text=text, max_seq_len=args.max_seq_length)
        examples.append((result['input_ids'], result['token_type_ids']))

    # Seperates data into some batches.
    batches = [
        examples[i:i + args.batch_size]
        for i in range(0, len(examples), args.batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.sigmoid(logits).numpy()
        confidence = []
        for prob in probs:
            labels = []
            for i, p in enumerate(prob):
                if p > 0.5:
                    labels.append(i)
            results.append(labels)

    for idx, text in enumerate(data):
        label_name = [label_list[r] for r in results[idx]]
        print("input data:", text)

        level1 = []
        level2 = []
        for r in results[idx]:
            if r < 7:
                level1.append(label_list[r])
            else:
                level2.append(label_list[r])
        print('predicted result:')
        print('level 1 : {} level 2 : {}'.format(', '.join(level1),
                                                 ', '.join(level2)))
    return


if __name__ == "__main__":
    data = [
        "a high degree of uncertainty associated with the emission inventory for china tends to degrade the performance of chemical transport models in predicting pm2.5 concentrations especially on a daily basis. in this study a novel machine learning algorithm, geographically -weighted gradient boosting machine (gw-gbm), was developed by improving gbm through building spatial smoothing kernels to weigh the loss function. this modification addressed the spatial nonstationarity of the relationships between pm2.5 concentrations and predictor variables such as aerosol optical depth (aod) and meteorological conditions. gw-gbm also overcame the estimation bias of pm2.5 concentrations due to missing aod retrievals, and thus potentially improved subsequent exposure analyses. gw-gbm showed good performance in predicting daily pm2.5 concentrations (r-2 = 0.76, rmse = 23.0 g/m(3)) even with partially missing aod data, which was better than the original gbm model (r-2 = 0.71, rmse = 25.3 g/m(3)). on the basis of the continuous spatiotemporal prediction of pm2.5 concentrations, it was predicted that 95% of the population lived in areas where the estimated annual mean pm2.5 concentration was higher than 35 g/m(3), and 45% of the population was exposed to pm2.5 >75 g/m(3) for over 100 days in 2014. gw-gbm accurately predicted continuous daily pm2.5 concentrations in china for assessing acute human health effects. (c) 2017 elsevier ltd. all rights reserved.",
        "previous research exploring cognitive biases in bulimia nervosa suggests that attentional biases occur for both food-related and body-related cues. individuals with bulimia were compared to non-bulimic controls on an emotional-stroop task which contained both food-related and body-related cues. results indicated that bulimics (but not controls) demonstrated a cognitive bias for both food-related and body related cues. however, a discrepancy between the two cue-types was observed with body-related cognitive biases showing the most robust effects and food-related cognitive biases being the most strongly associated with the severity of the disorder. the results may have implications for clinical practice as bulimics with an increased cognitive bias for food-related cues indicated increased bulimic disorder severity. (c) 2016 elsevier ltd. all rights reserved.",
        "posterior reversible encephalopathy syndrome (pres) is a reversible clinical and neuroradiological syndrome which may appear at any age and characterized by headache, altered consciousness, seizures, and cortical blindness. the exact incidence is still unknown. the most commonly identified causes include hypertensive encephalopathy, eclampsia, and some cytotoxic drugs. vasogenic edema related subcortical white matter lesions, hyperintense on t2a and flair sequences, in a relatively symmetrical pattern especially in the occipital and parietal lobes can be detected on cranial mr imaging. these findings tend to resolve partially or completely with early diagnosis and appropriate treatment. here in, we present a rare case of unilateral pres developed following the treatment with pazopanib, a testicular tumor vascular endothelial growth factor (vegf) inhibitory agent."
    ]
    label_list = get_wos_label_list()
    predict(data, label_list)
