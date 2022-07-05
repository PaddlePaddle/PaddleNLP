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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evaluates model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch['input_ids'], batch[
            'token_type_ids'], batch['labels']
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        losses.append(loss.numpy())
        metric.update(probs, labels)

    micro_f1_score, macro_f1_score = metric.accumulate()
    logger.info("loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f" %
                (np.mean(losses), micro_f1_score, macro_f1_score))
    model.train()
    metric.reset()

    return micro_f1_score, macro_f1_score


def preprocess_function(examples, tokenizer, max_seq_length, label_list, depth):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
        
    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    result = tokenizer(text=examples["sentence"], max_seq_len=max_seq_length)
    # One-Hot label
    labels = []
    layers = [examples["level {}".format(d + 1)] for d in range(depth)]
    shape = [len(layer) for layer in layers]
    offsets = [0] * len(shape)
    has_next = True
    while has_next:
        l = ''
        for i, off in enumerate(offsets):
            if l == '':
                l = layers[i][off]
            else:
                l += '--{}'.format(layers[i][off])
            if l in label_list and label_list[l] not in labels:
                labels.append(label_list[l])
        for i in range(len(shape) - 1, -1, -1):
            if offsets[i] + 1 >= shape[i]:
                offsets[i] = 0
                if i == 0:
                    has_next = False
            else:
                offsets[i] += 1
                break

    result["labels"] = [
        float(1) if i in labels else float(0) for i in range(len(label_list))
    ]
    return result


def get_wos_label_list():
    """
    Return labels of the WOS.
    """
    return [
        'CS', 'ECE', 'Psychology', 'MAE', 'Civil', 'Medical', 'biochemistry',
        'CS--Computer vision', 'CS--Machine learning', 'CS--network security',
        'CS--Cryptography', 'CS--Operating systems', 'CS--Computer graphics',
        'CS--Image processing', 'CS--Parallel computing',
        'CS--Relational databases', 'CS--Software engineering',
        'CS--Distributed computing', 'CS--Structured Storage',
        'CS--Symbolic computation', 'CS--Algorithm design',
        'CS--Computer programming', 'CS--Data structures', 'CS--Bioinformatics',
        'ECE--Electricity', 'ECE--Lorentz force law',
        'ECE--Electrical circuits', 'ECE--Voltage law', 'ECE--Digital control',
        'ECE--System identification', 'ECE--Electrical network',
        'ECE--Microcontroller',
        'ECE--Electrical generator/Analog signal processing',
        'ECE--Electric motor', 'ECE--Satellite radio',
        'ECE--Control engineering', 'ECE--Signal-flow graph',
        'ECE--State space representation', 'ECE--PID controller',
        'ECE--Operational amplifier', 'Psychology--Prejudice',
        'Psychology--Social cognition', 'Psychology--Person perception',
        'Psychology--Nonverbal communication', 'Psychology--Prosocial behavior',
        'Psychology--Leadership', 'Psychology--Eating disorders',
        'Psychology--Depression', 'Psychology--Borderline personality disorder',
        'Psychology--Seasonal affective disorder', 'Medical--Schizophrenia',
        'Psychology--Antisocial personality disorder',
        'Psychology--Media violence', 'Psychology--Prenatal development',
        'Psychology--Child abuse', 'Psychology--Gender roles',
        'Psychology--False memories', 'Psychology--Attention',
        'Psychology--Problem-solving', 'MAE--computer-aided design',
        'MAE--Hydraulics', 'MAE--Manufacturing engineering',
        'MAE--Machine design', 'MAE--Fluid mechanics',
        'MAE--Internal combustion engine', 'MAE--Thermodynamics',
        'MAE--Materials Engineering', 'MAE--Strength of materials',
        'Civil--Ambient Intelligence', 'Civil--Geotextile',
        'Civil--Remote Sensing', 'Civil--Rainwater Harvesting',
        'Civil--Water Pollution', 'Civil--Suspension Bridge',
        'Civil--Stealth Technology', 'Civil--Green Building',
        'Civil--Solar Energy', 'Civil--Construction Management',
        'Civil--Smart Material', 'Medical--Addiction', 'Medical--Allergies',
        "Medical--Alzheimer's Disease", 'Medical--Ankylosing Spondylitis',
        'Medical--Anxiety', 'Medical--Asthma', 'Medical--Atopic Dermatitis',
        'Medical--Atrial Fibrillation', 'Medical--Autism', 'Medical--Skin Care',
        'Medical--Bipolar Disorder', 'Medical--Birth Control',
        "Medical--Children's Health", "Medical--Crohn's Disease",
        'Medical--Dementia', 'Medical--Diabetes', 'Medical--Weight Loss',
        'Medical--Digestive Health', 'Medical--Emergency Contraception',
        'Medical--Mental Health', 'Medical--Fungal Infection',
        'Medical--Headache', 'Medical--Healthy Sleep', 'Medical--Heart Disease',
        'Medical--Hepatitis C', 'Medical--Hereditary Angioedema',
        'Medical--HIV/AIDS', 'Medical--Hypothyroidism',
        'Medical--Idiopathic Pulmonary Fibrosis',
        'Medical--Irritable Bowel Syndrome', 'Medical--Kidney Health',
        'Medical--Low Testosterone', 'Medical--Lymphoma', 'Medical--Medicare',
        'Medical--Menopause', 'Medical--Migraine',
        'Medical--Multiple Sclerosis', 'Medical--Myelofibrosis',
        'Medical--Cancer', 'Medical--Osteoarthritis', 'Medical--Osteoporosis',
        'Medical--Overactive Bladder', 'Medical--Parenting',
        "Medical--Parkinson's Disease", 'Medical--Polycythemia Vera',
        'Medical--Psoriasis', 'Medical--Psoriatic Arthritis',
        'Medical--Rheumatoid Arthritis', 'Medical--Senior Health',
        'Medical--Smoking Cessation', 'Medical--Sports Injuries',
        'Medical--Sprains and Strains', 'Medical--Stress Management',
        'biochemistry--Molecular biology', 'biochemistry--Cell biology',
        'biochemistry--Human Metabolism', 'biochemistry--Immunology',
        'biochemistry--Genetics', 'biochemistry--Enzymology',
        'biochemistry--Polymerase chain reaction',
        'biochemistry--Northern blotting', 'biochemistry--Southern blotting'
    ]
