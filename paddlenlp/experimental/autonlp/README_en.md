# AutoNLP

[简体中文🀄](./README.md) |  **English**🌎

# Introduction

**The AutoNLP APIs are subjective to significant changes until formal release**

**AutoNLP** is an experimental project by PaddleNLP to democratize NLP for everyone. Delivering a successful NLP project is not easy, as it requires deep domain knowledge. Time after time, we have seen people struggle to make NLP work on their dataset, for their projects, which is why we are building **AutoNLP**. Compared with the traditional AutoML approach of massive paid compute for State-of-the-Art model performance, we have a different philosphy:


1. Instead of training State-of-the-Art models on huge datasets running on huge clusters, our goal is to deliver **decent models under limited compute**. We assume our users have a few GPUs at most and want to get decent models under 8 hours on their own in-house datasets. Note that you can get this level of compute for FREE on [Baidu AI Studio](https://aistudio.baidu.com/aistudio).
2. Our solution is **low-code** and enables you to train good models with a few lines of code but it won't be no code / drag and drop.
3. Leverage the **full-cycle capability** of PaddleNLP, We intent to **automate and abstract away** as much of NLP as possible, ranging from preprocessing to tokenizing, from finetuning to prompt tuning, from model compression to deloyment, etc.
4. Our work is and always will be **free and open-sourced**.

## Installation

Installing **AutoNLP** is very similar to installing PaddleNLP, with the only difference being the `[autonlp]` tag.

```
pip install -U paddlenlp[autonlp]
```

You can also get our latest work in the develop branch by cloning from our [GitHub](https://github.com/PaddlePaddle/PaddleNLP) and install from source via `pip install .[autonlp]`.

## Basic Usage

Since the only supported task is Text Classification for now, the following documentation are on the usage of **AutoTrainerForTextClassification**. You can also follow our AiStudio notebook for example.

### Constructor

`AutoTrainerForTextClassification` is the main class which you use to run model experiments and interact with the trained models You can construct it like the following:

```python
auto_trainer = AutoTrainerForTextClassification(
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    label_column="labels",
    text_column="sentence",
    language="Chinese",
    output_dir="temp"
)
```

Args:

- train_dataset (Dataset, required): Training dataset in the format of `paddle.io.Dataset`, must contains the 'text_column' and 'label_column' specified below
- eval_dataset (Dataset, required): Evaluation dataset in the format of `paddle.io.Dataset`, must contains the 'text_column' and 'label_column' specified below
- text_column (string, required): Name of the column that contains the input text.
- label_column (string, required): Name of the column that contains the target variable to predict.
- language (string, required): language of the text
- metric_for_best_model (string, optional): the name of the metrc for selecting the best model.
- greater_is_better (bool, optional): Whether better models should have a greater metric or not. Use in conjuction with `metric_for_best_model`.
- problem_type (str, optional): Select among ["multi_class", "multi_label"] based on the nature of your problem
- output_dir (str, optional): Output directory for the experiments, defaults to "autpnlp_results"
- verbosity: (int, optional): controls the verbosity of the run. Defaults to 1, which let the workers log to the driver.To reduce the amount of logs, use verbosity > 0 to set stop the workers from logging to the driver.


### Train

You can start training with the following command:

```python
auto_trainer.train(
    num_cpus=2,
    num_gpus=1,
    max_concurrent_trials=1,
    num_models=10,
    time_budget_s=60 * 10,
    verbosity=1
)
```
Args:

- num_models (int, required): number of model trials to run
- num_gpus (str, optional): number of GPUs to use for the job. By default, this is set based on detected GPUs.
- num_cpus (str, optional): number of CPUs to use for the job. By default, this is set based on virtual cores.
- max_concurrent_trials (int, optional): maximum number of trials to run concurrently. Must be non-negative. If None or 0, no limit will be applied. Defaults to None.
- time_budget_s: (int|float|datetime.timedelta, optional) global time budget in seconds after which all model trials are stopped.
- experiment_name: (str, optional): name of the experiment. Experiment log will be stored under `<output_dir>/<experiment_name>`. Defaults to UNIX timestamp.
- hp_overrides: (dict[str, Any], optional): Advanced users only. override the hyperparameters of every model candidate.  For example, {"TrainingArguments.max_steps": 5}.
- custom_model_candiates: (dict[str, Any], optional): Advanced users only. Run the user-provided model candidates instead of the default model candidated from PaddleNLP. See `._model_candidates` property as an example

### Evaluations and Examine Results

#### Examine Results

Once the experimenets conclude, you can examine the experiment results like the following, which prints a pandas DataFrame:

```
auto_trainer.show_training_results()
```

You can also find the experiment results under `<output_dir>/experiment_results.csv`. The identifier for the models produced by different experiments is `trial_id`, which you can find in the `DataFrame` or the csv file.

#### Load Previous Results

You can recover the experiment results from a previous run (including unfinished runs) like the following:

```python
auto_trainer.load("path/to/previous/results")
```

This enables you to use the `show_training_results` API to examine the results. Call `train()` again will override the previous results.

#### Custom Evaluations

To evaluate on datasets other than the evaluation dataset provided to `AutoTrainerForTextClassification` at construction, you can use the

```
auto_trainer.evaluate(
    trial_id="trial_123456",
    eval_dataset=new_eval_dataset
)
```

Args:
- trial_id (str, optional): specify the model to be evaluated through the `trial_id`. Defaults to the best model, ranked by `metric_for_best_model`
- eval_dataset (Dataset, optional): custom evaluation dataset and must contains the 'text_column' and 'label_column' fields. If not provided, defaults to the evaluation dataset used at construction



### Export and Inference

To export a model for later use, do:

```
auto_trainer.export(
    trial_id="trial_123456",
    export_path="different/path/to/store/the/model"
)
```

Args:
- export_path (str, required): the filepath for export
- trial_id (int, required): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`

We also provide a convenience method to directly convert a model to a Taskflow for inference:

```
taskflow = auto_trainer.to_taskflow()
taskflow("this is a test input")
```

Args:
- trial_id (int, required): use the `trial_id` to select the model to export. Defaults to the best model selected by `metric_for_best_model`
