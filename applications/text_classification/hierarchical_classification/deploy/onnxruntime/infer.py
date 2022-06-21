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

import argparse

import psutil
import paddle
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset

from predictor import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path_prefix",
                    type=str,
                    required=True,
                    help="The path prefix of inference model to be used.")
parser.add_argument("--model_name_or_path",
                    default="ernie-2.0-base-en",
                    type=str,
                    help="The directory or name of model.")
parser.add_argument("--dataset",
                    default="wos",
                    type=str,
                    help="Dataset for hierarchical classfication tasks.")
parser.add_argument("--max_seq_length",
                    default=512,
                    type=int,
                    help="The maximum total input sequence length after "
                    "tokenization. Sequences longer than this will "
                    "be truncated, sequences shorter will be padded.")
parser.add_argument("--use_fp16",
                    action='store_true',
                    help="Whether to use fp16 inference, only "
                    "takes effect when deploying on gpu.")
parser.add_argument("--use_quantize",
                    action='store_true',
                    help="Whether to use quantization for acceleration,"
                    " only takes effect when deploying on cpu.")
parser.add_argument("--batch_size",
                    default=200,
                    type=int,
                    help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--num_threads",
                    default=psutil.cpu_count(logical=False),
                    type=int,
                    help="num_threads for cpu, only takes effect"
                    " when deploying on cpu.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--device_id',
                    default=0,
                    help="Select which gpu device to train model.")
parser.add_argument("--perf",
                    action='store_true',
                    help="Whether to compute the latency "
                    "and f1 score of the test set.")
args = parser.parse_args()


def predict():
    """
    Predicts the data labels.
    """
    predictor = Predictor(args)
    texts = [
        "a high degree of uncertainty associated with the emission inventory for china tends to degrade the performance of chemical transport models in predicting pm2.5 concentrations especially on a daily basis. in this study a novel machine learning algorithm, geographically -weighted gradient boosting machine (gw-gbm), was developed by improving gbm through building spatial smoothing kernels to weigh the loss function. this modification addressed the spatial nonstationarity of the relationships between pm2.5 concentrations and predictor variables such as aerosol optical depth (aod) and meteorological conditions. gw-gbm also overcame the estimation bias of pm2.5 concentrations due to missing aod retrievals, and thus potentially improved subsequent exposure analyses. gw-gbm showed good performance in predicting daily pm2.5 concentrations (r-2 = 0.76, rmse = 23.0 g/m(3)) even with partially missing aod data, which was better than the original gbm model (r-2 = 0.71, rmse = 25.3 g/m(3)). on the basis of the continuous spatiotemporal prediction of pm2.5 concentrations, it was predicted that 95% of the population lived in areas where the estimated annual mean pm2.5 concentration was higher than 35 g/m(3), and 45% of the population was exposed to pm2.5 >75 g/m(3) for over 100 days in 2014. gw-gbm accurately predicted continuous daily pm2.5 concentrations in china for assessing acute human health effects. (c) 2017 elsevier ltd. all rights reserved.",
        "previous research exploring cognitive biases in bulimia nervosa suggests that attentional biases occur for both food-related and body-related cues. individuals with bulimia were compared to non-bulimic controls on an emotional-stroop task which contained both food-related and body-related cues. results indicated that bulimics (but not controls) demonstrated a cognitive bias for both food-related and body related cues. however, a discrepancy between the two cue-types was observed with body-related cognitive biases showing the most robust effects and food-related cognitive biases being the most strongly associated with the severity of the disorder. the results may have implications for clinical practice as bulimics with an increased cognitive bias for food-related cues indicated increased bulimic disorder severity. (c) 2016 elsevier ltd. all rights reserved.",
        "posterior reversible encephalopathy syndrome (pres) is a reversible clinical and neuroradiological syndrome which may appear at any age and characterized by headache, altered consciousness, seizures, and cortical blindness. the exact incidence is still unknown. the most commonly identified causes include hypertensive encephalopathy, eclampsia, and some cytotoxic drugs. vasogenic edema related subcortical white matter lesions, hyperintense on t2a and flair sequences, in a relatively symmetrical pattern especially in the occipital and parietal lobes can be detected on cranial mr imaging. these findings tend to resolve partially or completely with early diagnosis and appropriate treatment. here in, we present a rare case of unilateral pres developed following the treatment with pazopanib, a testicular tumor vascular endothelial growth factor (vegf) inhibitory agent."
    ]
    predictor.predict(texts)

    if args.perf:

        texts = []
        labels = []
        test_ds = load_dataset(args.dataset, splits=["test"])
        for i in range(len(test_ds)):
            text = test_ds[i]['sentence']
            label = [
                float(1) if ii in test_ds[i]['label'] else float(0)
                for ii in range(len(test_ds.label_list))
            ]
            texts.append(text)
            labels.append(label)

        preprocess_result = predictor.preprocess(texts)

        # evaluate
        predictor.evaluate(preprocess_result, labels)

        # latency
        predictor.performance(preprocess_result)


if __name__ == "__main__":
    predict()
