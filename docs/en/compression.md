# PaddleNLP Model Compression API

**Table of Contents**
* [Introduction to Model Compression API Features](#introduction-to-model-compression-api-features)
* [Quick Start Examples for Three Major Scenarios](#quick-start-examples-for-three-major-scenarios)
* [Four Steps to Initiate Model Compression](#four-steps-to-initiate-model-compression)
    * [Step 1: Obtain Model Compression Parameters `compression_args`](#step-1-obtain-model-compression-parameters-compression_args)
    * [Step 2: Instantiate Trainer and Call `compress()`](#step-2-instantiate-trainer-and-call-compress)
        * [Introduction to Trainer Instantiation Parameters](#introduction-to-trainer-instantiation-parameters)
    * [Step 3: Implement Custom Evaluation Function to Adapt to Custom Compression Tasks](#step-3-implement-custom-evaluation-function-to-adapt-to-custom-compression-tasks)
    * [Step 4: Pass Parameters and Run Compression Script](#step-4-pass-parameters-and-run-compression-script)
        * [Introduction to `CompressionArguments` Parameters](#introduction-to-compressionarguments-parameters)
* [Model Evaluation and Deployment](#model-evaluation-and-deployment)
* [FAQ](#FAQ)
* [References](#References)

<a name="introduction-to-model-compression-api-features"></a>

## Introduction to Model Compression API Features

The PaddleNLP Model Compression API supports pruning and quantization of fine-tuned models on downstream ERNIE tasks to reduce model size, memory usage, computation, and enhance inference speed, thereby reducing deployment difficulty. The Model Compression API is effective and easy to use. Currently, the pruning feature supports the width-adaptive pruning strategy in DynaBERT; quantization supports static offline quantization methods (PTQ), quantization-aware training (QAT), and Embedding quantization. PTQ requires no training and only a small amount of calibration data to export a quantized model. QAT is similar to the training process of FP32 models and can generally achieve lossless precision. The Embedding quantization process is relatively simple, requiring neither training nor calibration data.

- **Effective**: It has been validated on tasks such as classification (including text classification, text matching, natural language inference, pronoun disambiguation, reading comprehension), sequence labeling, and extractive reading comprehension, achieving nearly lossless precision. For example, for models with 12L768H and 6L768H structures, pruning with a width retention ratio of 2/3 can achieve nearly lossless precision, and the inference speed of the pruned model can reach 1-2 times the original. The inference speed of a quantized 6L768H model can reach 2-3 times that before quantization.

- **Simple and Easy to Use**: Model compression tasks can be initiated in just a few simple steps.

##### ERNIE 3.0 Compression Results
As shown in the table below, the ERNIE 3.0-Medium (6-layer, 384-hidden, 12-heads) model achieves approximately 3x acceleration across three types of tasks (text classification, sequence labeling, extractive reading comprehension) after pruning + quantization, with an average precision loss across all tasks controlled within 0.5 (0.46).

|                            | TNEWS Performance | TNEWS Precision | MSRA_NER Performance | MSRA_NER Precision | CMRC2018 Performance | CMRC2018 Precision |
|----------------------------|-------------------|-----------------|----------------------|--------------------|----------------------|--------------------|
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x)     | 57.45           | 366.75(1.0x)         | 93.04              | 146.84(1.0x)         | 66.95              |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x)     | 56.99(-0.46)    | 889.33(2.4x)         | 92.70(-0.34)       | 348.84(2.4x)         | 66.32(-0.63)       |
| ERNIE 3.0-Medium+Pruned+FP32 | 1424.01(1.3x)    | 57.31(-0.14)    | 454.27(1.2x)         | 93.27(+0.23)       | 183.77(1.3x)         | 65.92(-1.03)       |
| ERNIE 3.0-Medium+Pruned+INT8 | 3635.48(3.2x)    | 57.26(-0.19)    | 1105.26(3.0x)        | 93.20(+0.16)       | 444.27(3.0x)         | 66.17(-0.78)       |

(The above data is from the [ERNIE 3.0 Performance Test Document](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/ernie-3.0/README.md#performance-test), which includes an introduction to the test environment)

##### UIE Compression Results

Taking the task of extracting information from reimbursement forms as an example, using...
`uie-base` undergoes fine-tuning to first obtain the original FP32 model, followed by further quantization using the QAT strategy. The quantized model achieves an F1 score 2.19 points higher than the original FP32 model.

| Models                  |  F1   |
|-------------------------|:-----:|
| uie-base+fine-tuning+FP32      | 91.93 |
| uie-base+fine-tuning+quantization+INT8 | 94.12 |


<a name="quick-start-examples-for-model-compression-in-three-major-scenarios"></a>

### Quick Start Examples for Model Compression in Three Major Scenarios

This project provides examples of using the compression API in three major scenarios: classification (including tasks such as text classification, text matching, natural language inference, and pronoun disambiguation), sequence labeling, and extractive reading comprehension. You can refer to the scripts [compress_seq_cls.py](../../slm/model_zoo/ernie-3.0/compress_seq_cls.py), [compress_token_cls.py](../../slm/model_zoo/ernie-3.0/compress_token_cls.py), and [compress_qa.py](../../slm/model_zoo/ernie-3.0/compress_qa.py) under the [ERNIE 3.0](../../slm/model_zoo/ernie-3.0) directory for each scenario. The startup methods are as follows:

```shell
# Classification task
# This script supports 7 classification tasks in CLUE, with different hyperparameters. Therefore, hyperparameter configurations for classification tasks are set using config.yml
python compress_seq_cls.py \
    --dataset "clue tnews"  \
    --model_name_or_path best_models/TNEWS  \
    --output_dir ./

# Sequence labeling task
python compress_token_cls.py \
    --dataset "msra_ner"  \
    --model_name_or_path best_models/MSRA_NER \
    --output_dir ./ \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.00005 \
    --remove_unused_columns False \
    --num_train_epochs 3

# Reading comprehension task
python compress_qa.py \
    --dataset "clue cmrc2018" \
    --model_name_or_path best_models/CMRC2018  \
    --output_dir ./ \
    --max_seq_length 512 \
    --learning_rate 0.00003 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --max_answer_length 50 \

```
The example code uses a built-in dataset from `datasets` for compression. If you wish to use a custom dataset for compression, please refer to the [datasets custom dataset loading documentation](https://huggingface.co/docs/datasets/loading).

<a name="four-step-model-compression"></a>

## Four-Step Model Compression

### Environment Dependencies

- paddlepaddle-gpu >=2.4.1
- paddlenlp >= 2.5
- paddleslim >= 2.4.0

The compression functionality in the model compression API relies on the latest `paddleslim` package. You can install it by running the following command:

```shell
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The usage of the model compression API is generally divided into four steps:

- Step 1: Use `PdArgumentParser` to parse hyperparameters passed from the command line to obtain the compression parameters `compression_args`;
- Step 2: Instantiate a Trainer and call the `compress()` compression API;
- Step 3: Implement custom evaluation functions and loss calculation functions (optional as needed) to adapt to custom compression tasks;
- Step 4: Pass parameters and run the compression script.

**Example Code**

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments

# Step1: Use `PdArgumentParser` to parse hyperparameters passed from the command line to obtain the compression parameters `compression_args`;
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()

# Step2: Instantiate a Trainer and call compress()
trainer = Trainer(
    model=model,
    args=compression_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    criterion=criterion)

# Step 3: If using built-in models and evaluation methods, there is no need to implement custom evaluation functions and loss calculation functions
trainer.compress()
```

```shell
# Step 4: Pass parameters and run the compression script
python compress.py \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4 \
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \
```

<a name="obtain-compression-parameters-compression_args"></a>

### Step 1: Obtain Compression Parameters `compression_args`

Use
The `PdArgumentParser` object parses hyperparameters obtained from the command line to generate `compression_args`, which are then passed to the `Trainer` object. The method to obtain `compression_args` is typically as follows:

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments

# Step 1: Use `PdArgumentParser` to parse hyperparameters passed from the command line to obtain compression parameters `compression_args`;
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()
```

<a name="instantiate-trainer-and-call-compress"></a>

### Step 2: Instantiate Trainer and Call Compress

<a name="trainer-instantiation-parameters-introduction"></a>

#### Trainer Instantiation Parameters Introduction

- **--model**: The model to be compressed. Currently supports models with similar structures such as ERNIE, BERT, RoBERTa, ERNIE-M, ELECTRA, ERNIE-Gram, PP-MiniLM, TinyBERT, which are fine-tuned on downstream tasks. When the pre-trained model is ERNIE, it needs to inherit from `ErniePretrainedModel`. For classification tasks, it can be obtained via `AutoModelForSequenceClassification.from_pretrained(model_name_or_path)`, where the `model_name_or_path` directory should contain the files model_config.json and model_state.pdparams;
- **--data_collator**: For all three types of tasks, PaddleNLP's predefined [DataCollator class](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/data/data_collator.py) can be used. The `data_collator` can perform operations such as `Pad` on the data. Refer to the [example code](../../slm/model_zoo/ernie-3.0/compress_seq_cls.py) for usage;
- **--train_dataset**: The training set used for pruning, which is task-specific data. For loading custom datasets, refer to the [documentation](https://huggingface.co/docs/datasets/loading). It can be None if pruning is not initiated;
- **--eval_dataset**: The evaluation set used for pruning training, also the calibration data for quantization, which is task-specific data. For loading custom datasets, refer to the [documentation](https://huggingface.co/docs/datasets/loading). It is a required parameter for the Trainer;
- **--tokenizer**: The `tokenizer` corresponding to the model `model`, which can be obtained using `AutoTokenizer.from_pretrained(model_name_or_path)`.
To obtain.
- **--criterion** The method for calculating the model's loss, which can be an `nn.Layer` object or a function, is used in `ofa_utils.py` to compute the model's loss for calculating gradients to determine the importance of neurons.

Example definition of the `criterion` function:

```python
# Supported form one:
def criterion(logits, labels):
    loss_fct = paddle.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = paddle.cast(start_ids, 'float32')
    end_ids = paddle.cast(end_ids, 'float32')
    loss_start = loss_fct(start_prob, start_ids)
    loss_end = loss_fct(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss

# Supported form two:
class CrossEntropyLossForSQuAD(paddle.nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits,
                                                        label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits,
                                                      label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss
```

Instantiate a `Trainer` object with the above parameters, and then directly call `compress()`. `compress()` will enter different branches based on the selected strategy to perform pruning or quantization.

**Example Code**
```python
def custom_evaluate(model, dataloader):
    # Initialize evaluation metrics
    metric = Accuracy()  # or any other metric you need
    model.eval()

    for batch in dataloader:
        # Forward pass
        inputs, labels = batch
        logits = model(inputs)

        # Update metric
        metric.update(logits, labels)

    # Return the computed metric
    return metric.compute()
```

### Step 3: Implement a Custom Evaluation Function to Adapt to Custom Compression Tasks

When using the DynaBERT pruning feature, if the model and metrics do not conform to the conditions in the table below, a custom evaluation function in the model compression API needs to be implemented.

Currently, the DynaBERT pruning feature only supports three types of built-in PaddleNLP classes such as SequenceClassification, and the corresponding built-in evaluators are Accuracy, F1, and Squad.

| Model class name | SequenceClassification | TokenClassification | QuestionAnswering |
|------------------|------------------------|---------------------|-------------------|
| Metrics          | Accuracy               | F1                  | Squad             |

Please note the following three conditions:

- If the model is a custom model, it needs to inherit from `XXXPretrainedModel`. For example, when the pre-trained model is ERNIE, it should inherit from `ErniePretrainedModel`. The model must support importing via `from_pretrained()` with only `pretrained_model_name_or_path` as a required parameter, and the `forward` function should return `logits` or a `tuple of logits`.

- If the model is custom or the dataset is particularly unique, and the loss calculation in the compression API does not meet the requirements, a custom `custom_evaluate` evaluation function needs to be implemented. This function must support both `paddleslim.nas.ofa.OFA` models and `paddle.nn.layer` models. Refer to the example code below.
    - The function should take `model` and `dataloader` as inputs and return the model's evaluation metric (a single float value).
    - Pass this function to the `custom_evaluate` parameter in the `compress()` method.

Example definition of the `custom_evaluate()` function:
```python
    import paddle
    from paddle.metric import Accuracy

    @paddle.no_grad()
    def evaluate_seq_cls(self, model, data_loader):
        metric = Accuracy()
        model.eval()
        metric.reset()
        for batch in data_loader:
            logits = model(input_ids=batch['input_ids'],
                           token_type_ids=batch['token_type_ids'])
            # Supports paddleslim.nas.ofa.OFA model and nn.layer model.
            if isinstance(model, paddleslim.nas.ofa.OFA):
                logits = logits[0]
            correct = metric.compute(logits, batch['labels'])
            metric.update(correct)
        res = metric.accumulate()
        logger.info("acc: %s, " % res)
        model.train()
        return res
```

When calling `compress()`, pass in this custom function:

```python
trainer.compress(custom_evaluate=evaluate_seq_cls)
```

<a name="Pass-parameters-and-run-compression-script"></a>

### Step 4: Pass parameters and run the compression script

This step primarily involves passing the parameters required for compression via the command line and initiating the compression script.

Compression initiation command:

**Example Code**

```shell
# Step 4: Run the compression script
python compress.py \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4 \
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \
```

Below, the hyperparameters that can be passed to the model compression initiation command will be introduced.

<a name="CompressionArguments-Parameter-Introduction"></a>

#### CompressionArguments Parameter Introduction

The parameters in `CompressionArguments` are partly specific to model compression functionality, while another part is inherited from `TrainingArguments`, which are hyperparameters that need to be set during compression training. The following will provide a detailed introduction.

**Common Parameters**

The parameters in the common parameters section are independent of specific compression strategies.

- **--strategy** Model compression strategy, currently supports
`'dynabert+qat+embeddings'`, `'dynabert+qat'`, `'dynabert+embeddings'`, `'dynabert+ptq'`, `'dynabert'`, `'ptq'`, and `'qat'`. Here, `'dynabert'` represents the width pruning strategy based on DynaBERT, `'qat'` denotes Quantization Aware Training, `'ptq'` indicates Post-Training Quantization, and `'embeddings'` signifies vocabulary quantization. The `--strategy` option supports selecting any reasonable combination of these strategies. The default is `'dynabert+ptq'`.

- **--output_dir**: Directory to save the model after compression.

- **--input_infer_model_path**: Path to the static graph model to be compressed. This parameter is intended to support the compression of static graph models. It can be ignored if not needed. The default is `None`.

- **--input_dtype**: Input type for the exported model, typically `int64` or `int32`. The default is `int64`.

**DynaBERT Pruning Parameters**

When the user employs DynaBERT pruning or PTQ quantization strategies (i.e., when the strategy includes 'dynabert' or 'qat'), the following optional parameters need to be provided:

- **--width_mult_list**: A search list for the retained width during pruning. For a 6-layer model, `3/4` is recommended, and for a 12-layer model, `2/3` is recommended. This indicates the retention ratio of the `q`, `k`, `v`, and `ffn` weight widths. For instance, if a 12-layer model originally has 12 attention heads, it will have only 9 attention heads after pruning. The default is `[3/4]`.

- **--per_device_train_batch_size**: Batch size per GPU/CPU core for pruning training. The default is 8.

- **--per_device_eval_batch_size**: Batch size per GPU/CPU core for pruning evaluation. The default is 8.

- **--num_train_epochs**: Number of epochs required for pruning training. The default is 3.0.

- **--max_steps**: If set to a positive number, it indicates the total number of training steps to execute. This overrides `num_train_epochs`.
- **--logging_steps**: The number of update steps between two logs. Default is 500.

- **--save_steps**: The number of steps for evaluating the model. Default is 100.

- **--optim**: The name of the optimizer used for pruning training, default is 'adamw'.

- **--learning_rate**: The initial learning rate for the optimizer used in pruning training. Default is 5e-05.

- **--weight_decay**: The weight decay value applied to all layers during pruning training, except for all bias and LayerNorm weights. Default is 0.0.

- **--adam_beta1**: The beta1 hyperparameter for the AdamW optimizer used in pruning training. Default is 0.9.

- **--adam_beta2**: The beta2 hyperparameter for the AdamW optimizer used in pruning training. Default is 0.999.

- **--adam_epsilon**: The epsilon hyperparameter for the AdamW optimizer used in pruning training. Default is 1e-8.

- **--max_grad_norm**: The maximum gradient norm (used for gradient clipping). Default is 1.0.

- **--lr_scheduler_type**: The learning rate scheduling strategy to use. Default is 'linear'.

- **--warmup_ratio**: The proportion of total training steps used for linear warmup from 0 to `learning_rate`. Default is 0.0.

- **--warmup_steps**: The number of steps used for linear warmup from 0 to `learning_rate`. Overrides the warmup_ratio parameter. Default is 0.

- **--seed**: The random seed set to ensure reproducibility across multiple runs. Default is 42.

- **--device**: The name of the device to run on. Supports cpu/gpu. Default is 'gpu'.

- **--remove_unused_columns**: Whether to remove unused field data from the Dataset. Default is True.

**Quantization Common Parameters**

**PTQ Quantization Parameters**

When the user employs the PTQ quantization strategy, the following optional parameters need to be provided:

- **--algo_list**: The list of quantization strategy searches, currently supporting `'KL'`, `'abs_max'`, `'min_max'`, `'avg'`, `'hist'`, `'mse'`, and `'emd'`. Different strategies calculate the quantization scale factor differently. It is recommended to provide multiple strategies to obtain multiple quantized models produced by various strategies, from which the optimal model can be selected. ERNIE models are recommended to use strategies like `'hist'`, `'mse'`, `'KL'`, and `'emd'`. Default is ['mse', 'KL'].

- **--batch_num_list**: The hyperparameter search list for batch_nums, where batch_nums indicates the number of batches required for sampling. The total amount of calibration data is batch_size * batch_nums. If batch_num is None, all data provided by the data loader will be used as calibration data. Default is [1].

- **--batch_size_list**: The search list for the batch_size of calibration samples. Bigger is not always better; it is also a hyperparameter. It is recommended to provide multiple calibration sample sizes, and the optimal model can be selected from multiple quantized models. Default is `[4]`.

- **--weight_quantize_type**: The quantization type for weights, supporting `'abs_max'` and `'channel_wise_abs_max'`.
Two methods. Typically, 'channel_wise_abs_max' is used, as this method usually results in a model with higher accuracy.

- **activation_quantize_type**: The quantization type for activation tensors. Supports 'abs_max', 'range_abs_max', and 'moving_average_abs_max'. In the 'ptq' strategy, the default is 'range_abs_max'.

- **--round_type**: The method for converting weight values from FP32 to INT8. Currently supports `'round'` and '[adaround](https://arxiv.org/abs/2004.10568)', with the default being `'round'`.

- **--bias_correction**: If set to True, it indicates the use of [bias correction](https://arxiv.org/abs/1810.05723) functionality, with the default being False.

**QAT Quantization Parameters**

When the user employs the QAT quantization strategy, in addition to setting the above training-related parameters, the following optional parameters can also be provided:

- **--weight_quantize_type**: The quantization type for weights, supporting both `'abs_max'` and `'channel_wise_abs_max'`. Typically, 'channel_wise_abs_max' is used, as this method usually results in a model with higher accuracy.

- **activation_quantize_type**: The quantization type for activation tensors. Supports 'abs_max', 'range_abs_max', and 'moving_average_abs_max'. In the 'qat' strategy, it defaults to 'moving_average_abs_max'.

- **use_pact**: Whether to use the PACT quantization strategy, which is an improvement over the standard method. Refer to the paper [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085). When enabled, it results in higher accuracy, with the default being True.

- **moving_rate**: The decay coefficient in the 'moving_average_abs_max' quantization method, with a default value of 0.9.

<a name="Model Evaluation and Deployment"></a>

## Model Evaluation and Deployment

Pruned and quantized models can no longer be loaded using `from_pretrained`.
Importing for prediction is not sufficient; instead, you must use Paddle deployment tools to complete the prediction.

For deploying the compressed model, refer to the [deployment documentation](../../slm/model_zoo/ernie-3.0/deploy).

### Python Deployment

Server-side deployment can begin here. You may refer to [seq_cls_infer.py](../../slm/model_zoo/ernie-3.0/deploy/python/seq_cls_infer.py) or [token_cls_infer.py](../../slm/model_zoo/ernie-3.0/deploy/python/token_cls_infer.py) to write your own prediction script. Follow the instructions in the [Python Deployment Guide](../../slm/model_zoo/ernie-3.0/deploy/python/README.md) to set up the prediction environment, evaluate the accuracy of the compressed model, conduct performance testing, and deploy.

<a name="service-deployment"></a>

### Service Deployment

- [FastDeploy ERNIE 3.0 Model Serving Deployment Example](../../slm/model_zoo/ernie-3.0/deploy/serving/README.md)
- [Service Deployment Based on PaddleNLP SimpleServing](../../slm/model_zoo/ernie-3.0/deploy/simple_serving/README.md)

### Mobile Deployment

<a name="FAQ"></a>

## FAQ

**Q: Does model compression require data?**

A: DynaBERT pruning and quantization-aware training (QAT) require the training set for training and the validation set for evaluation, similar to fine-tuning. Static offline quantization (PTQ) only requires the validation set (with a low sample size requirement, typically 4-16 samples may suffice).

**Q: The example code uses built-in datasets. How can I use my own data?**

A: You can refer to the UIE example or the [datasets custom dataset loading documentation](https://huggingface.co/docs/datasets/loading).

**Q: Can the compressed model continue training?**

A: Model compression is primarily for inference acceleration, so compressed models are static graph (prediction) models and cannot be imported for continued training using the `from_pretrained()` API.

**Q: How to choose between pruning and quantization?**

A: You can set the parameter `--strategy` to choose the compression strategy. By default, both pruning and quantization are selected, with pruning preceding quantization. Currently, the pruning strategy involves a training process that requires downstream task training data, and the training time depends on the data volume of the downstream task, comparable to fine-tuning. Static offline quantization does not require additional training and is faster; generally, quantization offers more significant acceleration than pruning. It is recommended to choose both pruning and quantization, as this may yield better results than quantization alone in some cases.

**Q: Is there a training process in pruning?**

A: DynaBERT pruning is similar to the distillation process and involves hyperparameters used during model training. For convenience, you can directly use the best hyperparameters from fine-tuning. To further improve accuracy, you can perform a Grid Search on hyperparameters such as `batch_size`, `learning_rate`, and `epoch`.

**Q: Why does using a `TensorDataset` object for quantization result in an error?**

A: When using quantization, the `eval_dataset` cannot be a `TensorDataset` object because the quantization function executes in static graph mode internally, and `TensorDataset`
Can only be used under dynamic graphs; using both simultaneously will cause errors.

<a name="References"></a>

## References
- Hou L, Huang Z, Shang L, Jiang X, Chen X, and Liu Q. DynaBERT: Dynamic BERT with Adaptive Width and Depth[J]. arXiv preprint arXiv:2004.04037, 2020.

- Cai H, Gan C, Wang T, Zhang Z, and Han S. Once for all: Train one network and specialize it for efficient deployment[J]. arXiv preprint arXiv:1908.09791, 2020.

- Wu H, Judd P, Zhang X, Isaev M, and Micikevicius P. Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation[J]. arXiv preprint arXiv:2004.09602v1, 2020.
