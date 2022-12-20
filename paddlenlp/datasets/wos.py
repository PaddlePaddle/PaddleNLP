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

# Copyright (c) 2017 Kamran Kowsari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this dataset and associated documentation files (the "Dataset"), to deal
# in the dataset without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Dataset, and to permit persons to whom the dataset is
# furnished to do so, subject to the following conditions:

import collections
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.datasets import DatasetBuilder

__all__ = ["WOS"]


class WOS(DatasetBuilder):
    """
    Web of Science(WOS) dataset contains abstracts of published papers from Web of Science.
    More information please refer to 'https://data.mendeley.com/datasets/9rw3vkcfy4/2'.
    """

    lazy = False
    URL = "https://bj.bcebos.com/paddlenlp/datasets/wos.tar.gz"
    MD5 = "15c8631ed6a474f471f480c31a6bbcda"
    META_INFO = collections.namedtuple("META_INFO", ("file", "md5"))
    SPLITS = {
        "train": META_INFO(os.path.join("wos", "train.tsv"), "e0153a1ef502235edf2bb138afcfef99"),
        "dev": META_INFO(os.path.join("wos", "dev.tsv"), "fcfc283349b353c3e1123fdd20429de9 "),
        "test": META_INFO(os.path.join("wos", "test.tsv"), "6fe2068aada7f17220d521dd11c73aee"),
    }

    def _get_data(self, mode, **kwargs):
        """Check and download Dataset"""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):

            get_path_from_url(self.URL, default_root, self.MD5)

        return fullname

    def _read(self, filename, *args):

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.split("\t")

                example = {"sentence": line_stripped[0].strip()}
                for i in range(len(line_stripped) - 1):
                    example["level {}".format(i + 1)] = line_stripped[i + 1].strip().split(",")

                yield example

    def get_labels(self):
        """
        Return labels of the WOS.
        """
        return [
            "CS",
            "ECE",
            "Psychology",
            "MAE",
            "Civil",
            "Medical",
            "biochemistry",
            "CS##Computer vision",
            "CS##Machine learning",
            "CS##network security",
            "CS##Cryptography",
            "CS##Operating systems",
            "CS##Computer graphics",
            "CS##Image processing",
            "CS##Parallel computing",
            "CS##Relational databases",
            "CS##Software engineering",
            "CS##Distributed computing",
            "CS##Structured Storage",
            "CS##Symbolic computation",
            "CS##Algorithm design",
            "CS##Computer programming",
            "CS##Data structures",
            "CS##Bioinformatics",
            "ECE##Electricity",
            "ECE##Lorentz force law",
            "ECE##Electrical circuits",
            "ECE##Voltage law",
            "ECE##Digital control",
            "ECE##System identification",
            "ECE##Electrical network",
            "ECE##Microcontroller",
            "ECE##Electrical generator/Analog signal processing",
            "ECE##Electric motor",
            "ECE##Satellite radio",
            "ECE##Control engineering",
            "ECE##Signal-flow graph",
            "ECE##State space representation",
            "ECE##PID controller",
            "ECE##Operational amplifier",
            "Psychology##Prejudice",
            "Psychology##Social cognition",
            "Psychology##Person perception",
            "Psychology##Nonverbal communication",
            "Psychology##Prosocial behavior",
            "Psychology##Leadership",
            "Psychology##Eating disorders",
            "Psychology##Depression",
            "Psychology##Borderline personality disorder",
            "Psychology##Seasonal affective disorder",
            "Medical##Schizophrenia",
            "Psychology##Antisocial personality disorder",
            "Psychology##Media violence",
            "Psychology##Prenatal development",
            "Psychology##Child abuse",
            "Psychology##Gender roles",
            "Psychology##False memories",
            "Psychology##Attention",
            "Psychology##Problem-solving",
            "MAE##computer-aided design",
            "MAE##Hydraulics",
            "MAE##Manufacturing engineering",
            "MAE##Machine design",
            "MAE##Fluid mechanics",
            "MAE##Internal combustion engine",
            "MAE##Thermodynamics",
            "MAE##Materials Engineering",
            "MAE##Strength of materials",
            "Civil##Ambient Intelligence",
            "Civil##Geotextile",
            "Civil##Remote Sensing",
            "Civil##Rainwater Harvesting",
            "Civil##Water Pollution",
            "Civil##Suspension Bridge",
            "Civil##Stealth Technology",
            "Civil##Green Building",
            "Civil##Solar Energy",
            "Civil##Construction Management",
            "Civil##Smart Material",
            "Medical##Addiction",
            "Medical##Allergies",
            "Medical##Alzheimer's Disease",
            "Medical##Ankylosing Spondylitis",
            "Medical##Anxiety",
            "Medical##Asthma",
            "Medical##Atopic Dermatitis",
            "Medical##Atrial Fibrillation",
            "Medical##Autism",
            "Medical##Skin Care",
            "Medical##Bipolar Disorder",
            "Medical##Birth Control",
            "Medical##Children's Health",
            "Medical##Crohn's Disease",
            "Medical##Dementia",
            "Medical##Diabetes",
            "Medical##Weight Loss",
            "Medical##Digestive Health",
            "Medical##Emergency Contraception",
            "Medical##Mental Health",
            "Medical##Fungal Infection",
            "Medical##Headache",
            "Medical##Healthy Sleep",
            "Medical##Heart Disease",
            "Medical##Hepatitis C",
            "Medical##Hereditary Angioedema",
            "Medical##HIV/AIDS",
            "Medical##Hypothyroidism",
            "Medical##Idiopathic Pulmonary Fibrosis",
            "Medical##Irritable Bowel Syndrome",
            "Medical##Kidney Health",
            "Medical##Low Testosterone",
            "Medical##Lymphoma",
            "Medical##Medicare",
            "Medical##Menopause",
            "Medical##Migraine",
            "Medical##Multiple Sclerosis",
            "Medical##Myelofibrosis",
            "Medical##Cancer",
            "Medical##Osteoarthritis",
            "Medical##Osteoporosis",
            "Medical##Overactive Bladder",
            "Medical##Parenting",
            "Medical##Parkinson's Disease",
            "Medical##Polycythemia Vera",
            "Medical##Psoriasis",
            "Medical##Psoriatic Arthritis",
            "Medical##Rheumatoid Arthritis",
            "Medical##Senior Health",
            "Medical##Smoking Cessation",
            "Medical##Sports Injuries",
            "Medical##Sprains and Strains",
            "Medical##Stress Management",
            "biochemistry##Molecular biology",
            "biochemistry##Cell biology",
            "biochemistry##Human Metabolism",
            "biochemistry##Immunology",
            "biochemistry##Genetics",
            "biochemistry##Enzymology",
            "biochemistry##Polymerase chain reaction",
            "biochemistry##Northern blotting",
            "biochemistry##Southern blotting",
        ]
