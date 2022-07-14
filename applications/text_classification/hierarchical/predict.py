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

parser = argparse.ArgumentParser()
parser.add_argument("--params_path",
                    default="./checkpoint/",
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
parser.add_argument("--depth",
                    type=int,
                    default=2,
                    help="The maximum level of hierarchy")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="The dataset directory including"
                    "data.txt and label.txt files.")
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
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)

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

        print("input data:", text)

        hierarchical_labels = {d: [] for d in range(args.depth)}

        for r in results[idx]:
            for i, l in enumerate(label_list[r].split('##')):
                if l not in hierarchical_labels[i]:
                    hierarchical_labels[i].append(l)
        print('predicted result:')
        for d in range(args.depth):
            print('level {}: {}'.format(d + 1,
                                        ', '.join(hierarchical_labels[d])))
        print('----------------------------')
    return


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
