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
parser.add_argument("--depth",
                    type=int,
                    default=2,
                    help="The maximum level of hierarchy")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="The dataset directory including "
                    "data.txt, label.txt, test.txt(optional,"
                    "if evaluate the performance).")
parser.add_argument("--perf_dataset",
                    choices=['dev', 'test'],
                    default='test',
                    type=str,
                    help="evaluate the performance on"
                    "dev dataset or test dataset")
args = parser.parse_args()


def predict(data, label_list):
    """
    Predicts the data labels.
    Args:

        data (obj:`List`): The processed data whose each element is one sequence.
        label_map(obj:`List`): The label id (key) to label str (value) map.
 
    """
    predictor = Predictor(args, label_list)
    predictor.predict(data)

    if args.perf:

        if args.dataset_dir is not None:
            eval_dir = os.path.join(args.dataset_dir,
                                    "{}.txt".format(args.perf_dataset))
            eval_ds = load_dataset("wos", data_files=(eval_dir))
        else:
            eval_ds = load_dataset(args.dataset, splits=[args.perf_dataset])

        texts, labels = predictor.get_text_and_label(eval_ds)

        preprocess_result = predictor.preprocess(texts)

        # evaluate
        predictor.evaluate(preprocess_result, labels)

        # latency
        predictor.performance(preprocess_result)


if __name__ == "__main__":

    if args.dataset_dir is not None:
        data_dir = os.path.join(args.dataset_dir, "data.txt")
        label_dir = os.path.join(args.dataset_dir, "label.txt")

        data = []
        label_list = []

        with open(data_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data.append(line.strip())
        f.close()

        with open(label_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())
        f.close()

    else:
        data = [
            "a high degree of uncertainty associated with the emission inventory for china tends to degrade the performance of chemical transport models in predicting pm2.5 concentrations especially on a daily basis. in this study a novel machine learning algorithm, geographically -weighted gradient boosting machine (gw-gbm), was developed by improving gbm through building spatial smoothing kernels to weigh the loss function. this modification addressed the spatial nonstationarity of the relationships between pm2.5 concentrations and predictor variables such as aerosol optical depth (aod) and meteorological conditions. gw-gbm also overcame the estimation bias of pm2.5 concentrations due to missing aod retrievals, and thus potentially improved subsequent exposure analyses. gw-gbm showed good performance in predicting daily pm2.5 concentrations (r-2 = 0.76, rmse = 23.0 g/m(3)) even with partially missing aod data, which was better than the original gbm model (r-2 = 0.71, rmse = 25.3 g/m(3)). on the basis of the continuous spatiotemporal prediction of pm2.5 concentrations, it was predicted that 95% of the population lived in areas where the estimated annual mean pm2.5 concentration was higher than 35 g/m(3), and 45% of the population was exposed to pm2.5 >75 g/m(3) for over 100 days in 2014. gw-gbm accurately predicted continuous daily pm2.5 concentrations in china for assessing acute human health effects. (c) 2017 elsevier ltd. all rights reserved.",
            "previous research exploring cognitive biases in bulimia nervosa suggests that attentional biases occur for both food-related and body-related cues. individuals with bulimia were compared to non-bulimic controls on an emotional-stroop task which contained both food-related and body-related cues. results indicated that bulimics (but not controls) demonstrated a cognitive bias for both food-related and body related cues. however, a discrepancy between the two cue-types was observed with body-related cognitive biases showing the most robust effects and food-related cognitive biases being the most strongly associated with the severity of the disorder. the results may have implications for clinical practice as bulimics with an increased cognitive bias for food-related cues indicated increased bulimic disorder severity. (c) 2016 elsevier ltd. all rights reserved.",
            "posterior reversible encephalopathy syndrome (pres) is a reversible clinical and neuroradiological syndrome which may appear at any age and characterized by headache, altered consciousness, seizures, and cortical blindness. the exact incidence is still unknown. the most commonly identified causes include hypertensive encephalopathy, eclampsia, and some cytotoxic drugs. vasogenic edema related subcortical white matter lesions, hyperintense on t2a and flair sequences, in a relatively symmetrical pattern especially in the occipital and parietal lobes can be detected on cranial mr imaging. these findings tend to resolve partially or completely with early diagnosis and appropriate treatment. here in, we present a rare case of unilateral pres developed following the treatment with pazopanib, a testicular tumor vascular endothelial growth factor (vegf) inhibitory agent."
        ]
        label_list = [
            'CS', 'ECE', 'Psychology', 'MAE', 'Civil', 'Medical',
            'biochemistry', 'CS##Computer vision', 'CS##Machine learning',
            'CS##network security', 'CS##Cryptography', 'CS##Operating systems',
            'CS##Computer graphics', 'CS##Image processing',
            'CS##Parallel computing', 'CS##Relational databases',
            'CS##Software engineering', 'CS##Distributed computing',
            'CS##Structured Storage', 'CS##Symbolic computation',
            'CS##Algorithm design', 'CS##Computer programming',
            'CS##Data structures', 'CS##Bioinformatics', 'ECE##Electricity',
            'ECE##Lorentz force law', 'ECE##Electrical circuits',
            'ECE##Voltage law', 'ECE##Digital control',
            'ECE##System identification', 'ECE##Electrical network',
            'ECE##Microcontroller',
            'ECE##Electrical generator/Analog signal processing',
            'ECE##Electric motor', 'ECE##Satellite radio',
            'ECE##Control engineering', 'ECE##Signal-flow graph',
            'ECE##State space representation', 'ECE##PID controller',
            'ECE##Operational amplifier', 'Psychology##Prejudice',
            'Psychology##Social cognition', 'Psychology##Person perception',
            'Psychology##Nonverbal communication',
            'Psychology##Prosocial behavior', 'Psychology##Leadership',
            'Psychology##Eating disorders', 'Psychology##Depression',
            'Psychology##Borderline personality disorder',
            'Psychology##Seasonal affective disorder', 'Medical##Schizophrenia',
            'Psychology##Antisocial personality disorder',
            'Psychology##Media violence', 'Psychology##Prenatal development',
            'Psychology##Child abuse', 'Psychology##Gender roles',
            'Psychology##False memories', 'Psychology##Attention',
            'Psychology##Problem-solving', 'MAE##computer-aided design',
            'MAE##Hydraulics', 'MAE##Manufacturing engineering',
            'MAE##Machine design', 'MAE##Fluid mechanics',
            'MAE##Internal combustion engine', 'MAE##Thermodynamics',
            'MAE##Materials Engineering', 'MAE##Strength of materials',
            'Civil##Ambient Intelligence', 'Civil##Geotextile',
            'Civil##Remote Sensing', 'Civil##Rainwater Harvesting',
            'Civil##Water Pollution', 'Civil##Suspension Bridge',
            'Civil##Stealth Technology', 'Civil##Green Building',
            'Civil##Solar Energy', 'Civil##Construction Management',
            'Civil##Smart Material', 'Medical##Addiction', 'Medical##Allergies',
            "Medical##Alzheimer's Disease", 'Medical##Ankylosing Spondylitis',
            'Medical##Anxiety', 'Medical##Asthma', 'Medical##Atopic Dermatitis',
            'Medical##Atrial Fibrillation', 'Medical##Autism',
            'Medical##Skin Care', 'Medical##Bipolar Disorder',
            'Medical##Birth Control', "Medical##Children's Health",
            "Medical##Crohn's Disease", 'Medical##Dementia',
            'Medical##Diabetes', 'Medical##Weight Loss',
            'Medical##Digestive Health', 'Medical##Emergency Contraception',
            'Medical##Mental Health', 'Medical##Fungal Infection',
            'Medical##Headache', 'Medical##Healthy Sleep',
            'Medical##Heart Disease', 'Medical##Hepatitis C',
            'Medical##Hereditary Angioedema', 'Medical##HIV/AIDS',
            'Medical##Hypothyroidism', 'Medical##Idiopathic Pulmonary Fibrosis',
            'Medical##Irritable Bowel Syndrome', 'Medical##Kidney Health',
            'Medical##Low Testosterone', 'Medical##Lymphoma',
            'Medical##Medicare', 'Medical##Menopause', 'Medical##Migraine',
            'Medical##Multiple Sclerosis', 'Medical##Myelofibrosis',
            'Medical##Cancer', 'Medical##Osteoarthritis',
            'Medical##Osteoporosis', 'Medical##Overactive Bladder',
            'Medical##Parenting', "Medical##Parkinson's Disease",
            'Medical##Polycythemia Vera', 'Medical##Psoriasis',
            'Medical##Psoriatic Arthritis', 'Medical##Rheumatoid Arthritis',
            'Medical##Senior Health', 'Medical##Smoking Cessation',
            'Medical##Sports Injuries', 'Medical##Sprains and Strains',
            'Medical##Stress Management', 'biochemistry##Molecular biology',
            'biochemistry##Cell biology', 'biochemistry##Human Metabolism',
            'biochemistry##Immunology', 'biochemistry##Genetics',
            'biochemistry##Enzymology',
            'biochemistry##Polymerase chain reaction',
            'biochemistry##Northern blotting', 'biochemistry##Southern blotting'
        ]

    predict(data, label_list)
