# RGL: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning

This is the implementation of the paper [RGL: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning](https://aclanthology.org/2022.findings-naacl.81/).

**************************** Updates *****************************

2022-07-11: Our training code has been released.

2022-04-08: Our paper has been accepted to Findings of [NAACL 2022](https://aclanthology.org/2022.findings-naacl.81/)!

# Overview

<h2 align="center">
<img align="center"  src="https://user-images.githubusercontent.com/25607475/178176845-c559b07f-5278-432d-a4d8-ed9bd74d393c.png" alt="overview" width = "410" height = "400">
</h2>

We propose a simple yet effective Relation Graph augmented Learning RGL method that can obtain better performance in few-shot natural language understanding tasks.

RGL constructs a relation graph based on the label consistency between samples in the same batch, and learns to solve the resultant node classification and link prediction problems of the relation graphs. In this way, RGL fully exploits the limited supervised information, which can boost the tuning effectiveness.

# Prepare the data

We evaluate on the GLUE variant for few-shot learning in the paper, including SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, MNLI-mm, SNLI, QNLI, RTE, MRPC, QQP and STS-B. Please download the [datasets](https://paddlenlp.bj.bcebos.com/datasets/k-shot-glue/rgl-k-shot.zip) and extract the data files to the path ``./data/k-shot``.


# Experiments

The structure of the code:

```
├── scripts/
│   ├── run_pet.sh  # Script for PET
│   └── run_rgl.sh  # Script for RGL
├── template.py     # The parser for prompt template
├── verbalizer.py   # The mapping from labels to corresponding words
├── tokenizer.py    # The tokenizer wrapeer to conduct text truncation
├── utils.py        # The tools
└── rgl.py          # The training process of RGL
```

## How to define a template

We inspire from [OpenPrompt](https://github.com/thunlp/OpenPrompt/tree/main) and define template as a list of dictionary. The key of raw texts in datasets is `text`, and the corresponding value is the keyword of text in loaded dataset, where we use `text_a` to denote the first sentence in every example and `text_b` to denote the other sentences by default.

For example, given the template ``{'text':'text_a'} It was {'mask'}.`` and a sample text ``nothing happens , and it happens to flat characters .`` the input text will be ``nothing happens , and it happens to flat characters . It was <mask>.``


## Quick start

Run the following code for prompt-tuning.

```
export CUDA_VISIBLE_DEVICES=0
python rgl.py \
--output_dir ./checkpoints/ \
--dataset SST-2 \
--data_path ./data/k-hot/SST-2/16-13/ \
--max_seq_length 128 \
--max_steps 1000 \
--logging_step 10 \
--eval_step 100 \
--batch_size 4 \
--alpha 0.1 \
--seed 13 \
--learning_rate 1e-5 \
--template "{'text':'text_a'} It was {'mask'}." \
--verbalizer "{'0':'terrible','1':'great'}"
```

The configurations consist of:
- ``output_dir``: The directory to save model checkpoints.
- ``dataset``: The dataset name for few-shot learning.
- ``data_path``: The path to data files of ``dataset``.
- ``max_seq_length``: The maximum length of input text, including the prompt.
- ``max_steps``: The maximum steps for training.
- ``logging_step``: Print logs every ``logging_step``.
- ``eval_step``: Evaluate model every ``eval_step``.
- ``batch_size``: The number of samples per batch.
- ``alpha``: The weight of the loss proposed in RGL.
- ``seed``: Random seed.
- ``learning_rate``: The learning rate for tuning.
- ``template``: The template to define how to combine text data and prompt.
- ``verbalizer``: The verbalizer to map labels to words in vocabulary.


## Multiple runs for the best results

To reproduce our experiments, you can use the scripts to get the results under different settings. We have defined the templates and the verbalizers in both ``./script/run_pet.sh`` and ``./script/run_rgl.sh``. You can refer to these scripts for more details.

### Run PET

```
bash ./scripts/run_pet.sh SST-2 0
```

where ``SST-2`` specifies the dataset used for prompt-tuning and you can replace it with any other downloaded datasets in ``./data/k-shot/ ``. Besides, ``0`` refers to the gpu device id.

**NOTE**: The dataset name is case-sensitive to run the scripts.

### Run RGL

```
bash ./scripts/run_rgl.sh SST-2 0
```

Please see the descriptions above for the arguments.


# Citation

Please cite our paper if you use RGL in your work:
```
@inproceedings{wang-etal-2022-rgl,
    title = "{RGL}: A Simple yet Effective Relation Graph Augmented Prompt-based Tuning Approach for Few-Shot Learning",
    author = "Wang, Yaqing and
      Tian, Xin  and
      Xiong, Haoyi  and
      Li, Yueyang  and
      Chen, Zeyu  and
      Guo, Sheng  and
      Dou, Dejing",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.81",
    pages = "1078--1084",
}

```
