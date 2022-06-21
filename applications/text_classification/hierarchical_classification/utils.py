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


def preprocess_function(examples, tokenizer, max_seq_length, label_nums):
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
    result["labels"] = [
        float(1) if i in examples["label"] else float(0)
        for i in range(label_nums)
    ]
    return result


def get_wos_label_list():
    """
    Return labels of the WOS.
    """
    return [
        'CS', 'ECE', 'Psychology', 'MAE', 'Civil', 'Medical', 'biochemistry',
        'Computer vision', 'Machine learning', 'network security',
        'Cryptography', 'Operating systems', 'Computer graphics',
        'Image processing', 'Parallel computing', 'Relational databases',
        'Software engineering', 'Distributed computing', 'Structured Storage',
        'Symbolic computation', 'Algorithm design', 'Computer programming',
        'Data structures', 'Bioinformatics', 'Electricity', 'Lorentz force law',
        'Electrical circuits', 'Voltage law', 'Digital control',
        'System identification', 'Electrical network', 'Microcontroller',
        'Electrical generator', 'Electric motor', 'Satellite radio',
        'Control engineering', 'Signal', 'State space representation',
        'PID controller', 'Operational amplifier', 'Prejudice',
        'Social cognition', 'Person perception', 'Nonverbal communication',
        'Prosocial behavior', 'Leadership', 'Eating disorders', 'Depression',
        'Borderline personality disorder', 'Seasonal affective disorder',
        'Schizophrenia', 'Antisocial personality disorder', 'Media violence',
        'Prenatal development', 'Child abuse', 'Gender roles', 'False memories',
        'Attention', 'Problem', 'computer', 'Hydraulics',
        'Manufacturing engineering', 'Machine design', 'Fluid mechanics',
        'Internal combustion engine', 'Thermodynamics', 'Materials Engineering',
        'Strength of materials', 'Ambient Intelligence', 'Geotextile',
        'Remote Sensing', 'Rainwater Harvesting', 'Water Pollution',
        'Suspension Bridge', 'Stealth Technology', 'Green Building',
        'Solar Energy', 'Construction Management', 'Smart Material',
        'Addiction', 'Allergies', "Alzheimer's Disease",
        'Ankylosing Spondylitis', 'Anxiety', 'Asthma', 'Atopic Dermatitis',
        'Atrial Fibrillation', 'Autism', 'Skin Care', 'Bipolar Disorder',
        'Birth Control', "Children's Health", "Crohn's Disease", 'Dementia',
        'Diabetes', 'Weight Loss', 'Digestive Health',
        'Emergency Contraception', 'Mental Health', 'Fungal Infection',
        'Headache', 'Healthy Sleep', 'Heart Disease', 'Hepatitis C',
        'Hereditary Angioedema', 'HIV/AIDS', 'Hypothyroidism',
        'Idiopathic Pulmonary Fibrosis', 'Irritable Bowel Syndrome',
        'Kidney Health', 'Low Testosterone', 'Lymphoma', 'Medicare',
        'Menopause', 'Migraine', 'Multiple Sclerosis', 'Myelofibrosis',
        'Cancer', 'Osteoarthritis', 'Osteoporosis', 'Overactive Bladder',
        'Parenting', "Parkinson's Disease", 'Polycythemia Vera', 'Psoriasis',
        'Psoriatic Arthritis', 'Rheumatoid Arthritis', 'Senior Health',
        'Smoking Cessation', 'Sports Injuries', 'Sprains and Strains',
        'Stress Management', 'Molecular biology', 'Cell biology',
        'Human Metabolism', 'Immunology', 'Genetics', 'Enzymology',
        'Polymerase chain reaction', 'Northern blotting', 'Southern blotting'
    ]
