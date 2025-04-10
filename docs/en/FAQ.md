## PaddleNLP Frequently Asked Questions (Continuously Updated)

+ [【Selected】Top 5 NLP Questions](#NLP-精选)

  + [Q1.1 How to load my own local dataset to use PaddleNLP's features?](#1-1)
  + [Q1.2 PaddleNLP downloads built-in datasets and models to a default path. How can I change this path?](#1-2)
  + [Q1.3 How to save and load a trained model in PaddleNLP?](#1-3)
  + [Q1.4 When training samples are limited, what recommended methods can improve model performance?](#1-4)
  + [Q1.5 How to enhance model performance and increase QPS?](#1-5)

+ [【Theory】General NLP Questions](#NLP-通用问题)

  + [Q2.1 What are the methods to handle imbalanced data distribution?](#2-2)
  + [Q2.2 How many samples are generally needed when using a pre-trained model?](#2-3)

+ [【Practical】PaddleNLP Practical Issues](#PaddleNLP-实战问题)

  [Dataset and Data Processing](#数据问题)

  + [Q3.1 How to introduce an additional vocabulary when training a pre-trained model with my own dataset?](#3-1)

  [Model Training and Optimization](#训练调优问题)

  + [Q3.2 How to load my own pre-trained model to use PaddleNLP's features?](#4-1)
  + [Q3.3 If training is interrupted and needs to resume, how to ensure the learning rate and optimizer continue from the interruption point?](#4-2)
  + [Q3.4 How to freeze model gradients?](#4-3)
  + [Q3.5 How to print evaluation metrics during the eval phase and save model parameters at each epoch?](#4-4)
  + [Q3.6 If the training process unexpectedly exits or hangs, how should I troubleshoot?](#4-5)

  + [Q3.7 How to ensure consistent results in model validation and testing every time?](#4-6)
  + [Q3.8 How does the ERNIE model return outputs from intermediate layers?](#4-7)

  [Prediction Deployment](#部署问题)

  + [Q3.9 How to deploy a trained PaddleNLP model to a server?](#5-1)
  + [Q3.10 How to convert a static graph model to a dynamic graph model?](#5-2)

+ [Specific Models and Application Scenarios Consultation](#NLP-应用场景)
  + [Q4.1 【Lexical Analysis】For the LAC model, how to customize labels and continue training?](#6-1)
  + [Q4.2 In information extraction tasks, is it recommended to use a pre-trained model + CRF, and how to implement it?](#6-2)
  + [Q4.3 【Reading Comprehension】How to understand `batched=True` in the `map()` method of `MapDatasets`, and why must the `batched` parameter be set to `True` in reading comprehension tasks?](#6-3)
？](#6-3)
  + [Q4.4 【Semantic Matching】What is the difference between semantic indexing and semantic matching?](#6-4)
  + [Q4.5 【Word Tagging】How to customize and add named entities and corresponding word classes in the wordtag model?](#6-5)

+ [Other Usage Inquiries](#usage-inquiries)
  + [Q5.1 Error when using PaddleNLP with CUDA11?](#7-1)
  + [Q5.2 How to set parameters?](#7-2)
  + [Q5.3 Although the GPU version of Paddle can run on a CPU, is a GPU device necessary?](#7-3)
  + [Q5.4 How to specify whether to train the model using CPU or GPU?](#7-4)
  + [Q5.5 Are the prediction results of dynamic graph models and static graph models consistent?](#7-5)
  + [Q5.6 How to visualize acc, loss curves, model network structure diagrams, etc.?](#7-6)

<a name="NLP Selection"></a>

## ⭐️【Selection】Top 5 NLP Questions

<a name="1-1"></a>

##### Q1.1 How to load your own local dataset to use PaddleNLP's features?

**A:** By using PaddleNLP's `load_dataset`, `MapDataset`, and `IterDataset`, you can easily customize your own dataset. You are also welcome to contribute datasets to the PaddleNLP repo.

When creating a dataset from a local file, we **recommend** providing a reading function based on the format of the local dataset and passing it into `load_dataset()` to create the dataset.

```python
from paddlenlp.datasets import load_dataset

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # Skip column names
        next(f)
        for line in f:
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            yield {'tokens': words, 'labels': labels}

# data_path is a parameter for the read() method
map_ds = load_dataset(read, data_path='train.txt', lazy=False)
iter_ds = load_dataset(read, data_path='train.txt', lazy=True)
```

If you are accustomed to using `paddle.io.Dataset/IterableDataset` to create datasets, it is also supported. You can also create datasets from other Python objects such as `List` objects. For more details, please refer to the [official documentation - Custom Dataset](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html).

<a name="1-2"></a>

##### Q1.2 PaddleNLP downloads built-in datasets and models to a default path. How can the path be modified?

**A:** Built-in datasets and models are downloaded by default to
`$HOME/.paddlenlp/` directory allows downloading to a specified path by configuring environment variables:

(1) On Linux, set `export PPNLP_HOME="xxxx"`. Ensure the path does not contain Chinese characters.

(2) On Windows, similarly configure the environment variable PPNLP_HOME to a path without Chinese characters, then restart.

<a name="1-3"></a>

##### Q1.3 How to save and load a trained model in PaddleNLP?

**A:** (1) PaddleNLP Pre-trained Models

   Save:

```python
model.save_pretrained("./checkpoint")
tokenizer.save_pretrained("./checkpoint")
```

   Load:

```python
model.from_pretrained("./checkpoint")
tokenizer.from_pretrained("./checkpoint")
```

(2) Standard Models
   Save:

```python
emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams") # Save model parameters
```

   Load:
```python
emb = paddle.nn.Embedding(10, 10)
load_layer_state_dict = paddle.load("emb.pdparams") # Load model parameters
emb.set_state_dict(load_layer_state_dict) # Set model parameters
```
<a name="1-4"></a>

##### Q1.4 What recommended methods can improve model performance when the training samples are limited?

**A:** Increasing the number of training samples is the most direct approach. Additionally, you can perform warm start based on our open-source [pre-trained models](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers) and then fine-tune the model with a small dataset. Moreover, for scenarios like classification and matching, [few-shot learning](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/few_shot) can also yield good results.

<a name="1-5"></a>

##### Q1.5 How can model performance be improved to enhance QPS?

**A:** From an engineering perspective, for server-side deployment, you can use the [Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html) high-performance prediction engine for deployment. For GPU prediction of Transformer models, you can use the [FastGeneration](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/ops) feature provided by PaddleNLP for fast prediction, which integrates [NV FasterTransformer](https://github.com/NVIDIA/FasterTransformer) with enhanced functionality.

From a model strategy perspective, you can use model miniaturization techniques for model compression, such as model distillation and pruning, to achieve acceleration with smaller models. PaddleNLP integrates general small models like ERNIE-Tiny for downstream task fine-tuning. Additionally, PaddleNLP provides [model compression examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/model_compression), implementing methods like DynaBERT, TinyBERT, and MiniLM, which can be referenced for distillation and compression of your own models.

<a name="NLP 通用问题"></a>

## ⭐️【Theory Section】General NLP Issues

<a name="2-2"></a>

##### Q2.1 What methods can address imbalanced data distribution?

**A:** The following methods can optimize the issue of imbalanced class distribution:

(1) Under-sampling: Reduce the number of samples in over-represented classes to make the number of samples in each class similar.

(2) Over-sampling: Increase the number of samples in under-represented classes by duplicating samples to make the number of samples in each class similar.

(3) Adjusting classification threshold: Training a classifier with imbalanced data can bias the model towards the majority class, so instead of using 0.5 as the classification threshold, classify samples as the minority class even with low confidence.

(4) Cost-sensitive learning: For instance, set the `class_weight` parameter in the LR algorithm.

<a name="2-3"></a>

##### Q2.2 How many samples are generally needed when using pre-trained models?

**A:** It is difficult to define a specific number of samples as it depends on the specific task and data quality. If the data quality is good, classification and text matching tasks typically require data in the hundreds, while translation tasks require millions to train a robust model. If the sample size is small, consider data augmentation or few-shot learning.

<a name="PaddleNLP 实战问题"></a>

## ⭐️【Practical Section】PaddleNLP Practical Issues

<a name="数据问题"></a>

### Dataset and Data Processing

<a name="3-1"></a>

##### Q3.1 How to introduce additional vocabularies when training pre-trained models with your own dataset?

**A:** Pre-trained models usually come with a tokenizer and dictionary. For most Chinese pre-trained models, such as ERNIE-3.0, character-level input is used, and the tokenizer converts sentences into character-level forms, so the model cannot receive word-level input. If you wish to introduce additional vocabularies, you need to modify the tokenizer and dictionary of the pre-trained model. You can refer to this [blog](https://kexue.fm/archives/7758/comment-page-1#Tokenizer). Additionally, ensure that the embedding matrix includes embeddings for these new words.

Another approach is to use these dictionary words by masking them in the data for a masked language model's secondary pre-training. This way, the model trained through secondary training will include representations of the additional dictionary. Refer to [PaddleNLP Pre-training Data Process](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/ernie-1.0/).

Furthermore, some pre-trained models use word-level or mixed character-word level inputs, making it easier to introduce additional vocabularies. We will continue to enrich the pre-trained models in PaddleNLP.

<a name="训练调优问题"></a>

### Model Training and Optimization

<a name="4-1"></a>

##### Q3.2 How to load your own pre-trained model to use PaddleNLP's features?

**A:** Taking BERT as an example, if trained using PaddleNLP, through
Models saved using the `save_pretrained()` interface can be loaded with `from_pretrained()`:

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

If the above situation does not apply, you can load the model using the following method. Contributions to the PaddleNLP repository are also welcome.

(1) Load `BertTokenizer` and `BertModel`

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

(2) Use `save_pretrained()` to generate `model_config.json`, `tokenizer_config.json`, `model_state.pdparams`, and `vocab.txt` files, saving them to `./checkpoint`:

```python
tokenizer.save_pretrained("./checkpoint")
model.save_pretrained("./checkpoint")
```

(3) Modify the `model_config.json` and `tokenizer_config.json` configuration files to specify your own model, then load the model using `from_pretrained()`.

```python
tokenizer = BertTokenizer.from_pretrained("./checkpoint")
model = BertModel.from_pretrained("./checkpoint")
```

<a name="4-2"></a>

##### Q3.3 If training is interrupted and needs to resume with a warm start, how can you ensure the learning rate and optimizer continue iterating from the interruption point?

**A:**

(1) To fully restore the training state, first save the parameters of `lr`, `optimizer`, and `model`:

```python
paddle.save(lr_scheduler.state_dict(), "xxx_lr")
paddle.save(optimizer.state_dict(), "xxx_opt")
paddle.save(model.state_dict(), "xxx_para")
```

(2) Load the `lr`, `optimizer`, and `model` parameters to resume training:

```python
lr_scheduler.set_state_dict(paddle.load("xxxx_lr"))
optimizer.set_state_dict(paddle.load("xxx_opt"))
model.set_state_dict(paddle.load("xxx_para"))
```

<a name="4-3"></a>

##### Q3.4 How to freeze model gradients?

**A:**
There are several methods you can try:

(1) You can directly modify the internal code implementation of PaddleNLP, wrapping the sections where gradients need to be frozen with `paddle.no_grad()`.

The usage of `paddle.no_grad()` is as follows:
`forward()` freezing example:

```python
   # Method 1
   class Model(nn.Layer):
      def __init__(self, ...):
         ...

      def forward(self, ...):
         with paddle.no_grad():
            ...


   # Method 2
   class Model(nn.Layer):
      def __init__(self, ...):
         ...

      @paddle.no_grad()
      def forward(self, ...):
         ...
```

The use of `paddle.no_grad()` is not limited to the internal implementation of the model; it can also wrap external methods, such as:

```python
   @paddle.no_grad()
   def evaluation(...):
      ...

      model = Model(...)
      model.eval()

      ...
```

(2) Second method: Taking ERNIE as an example, set the `stop_gradient` of the tensor output by the model to True. You can use `register_forward_post_hook` to attempt as follows:

```python
   def forward_post_hook(layer, input, output):
      output.stop_gradient=True

   self.ernie.register_forward_post_hook(forward_post_hook)
```

(3) Third method: Process on the `optimizer`, `model.parameters` is a `List`, and you can filter accordingly by `name` to update/not update certain parameters. This method requires an overall understanding of the network structure's names, as the entity names of the network structure determine the parameter names, which poses a certain threshold for use:

```python
   [ p for p in model.parameters() if 'linear' not in p.name]  # Here you can filter out the linear layer, and the specific filtering strategy can be set as needed
```

<a name="4-4"></a>

##### Q3.5 How to print evaluation metrics during the eval phase and save model parameters at each epoch?

**A:** The PaddlePaddle main framework provides two methods for training and prediction. One method is to encapsulate the model using [paddle.Model()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html) and complete model training and prediction through high-level APIs such as `Model.fit()`, `Model.evaluate()`, `Model.predict()`, etc.; the other method is the conventional training method based on basic APIs.

(1) For the first method:

- We can set `paddle.Model.fit()`
 API's *eval_data* and *eval_freq* parameters are used to print model evaluation metrics during training: the *eval_data* parameter is an iterable validation dataset source, and the *eval_freq* parameter determines the evaluation frequency. When *eval_data* is provided, the default value of *eval_freq* is 1, meaning evaluation occurs once per epoch. Note: Before training, we need to pass the metrics parameter in the `Model.prepare()` interface to print model evaluation metrics during evaluation.

- Regarding model saving, we can set the *save_freq* parameter in `paddle.Model.fit()` to control the frequency of model saving: the default value of *save_freq* is 1, meaning the model is saved once per epoch.

(2) For the second method:

- We provide training and prediction scripts for common tasks in the examples directory of PaddleNLP, such as [GLUE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/benchmark/glue) and [SQuAD](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/machine_reading_comprehension/SQuAD).

- Developers can refer to the above scripts to develop custom training and prediction scripts.

<a name="4-5"></a>

##### Q3.6 What should be done if the training process unexpectedly exits or hangs?

**A:** Generally, first consider whether there is insufficient memory or GPU memory (if training with a GPU). You can reduce the batch size for training and evaluation.

Note that when reducing the batch size, the learning rate should also be reduced, typically adjusted proportionally.

<a name="4-6"></a>

##### Q3.7 How to ensure consistent results in model validation and testing?

**A:** Inconsistencies in results during validation and testing can generally be addressed by the following solutions:

(1) Ensure that the eval mode is set and that the seed settings related to data ensure data consistency.

(2) If it is a downstream task model, check whether all model parameters have been loaded. Directly using a pre-trained model like bert-base does not include task-related parameters, so ensure that the fine-tuned model is loaded; otherwise, task-related parameters will be randomly initialized, leading to randomness.

(3) Inconsistencies caused by some operators using the CUDNN backend can be avoided by setting environment variables. If the model uses CNN-related operators, you can set `FLAGS_cudnn_deterministic=True`. If the model uses RNN-related operators, you can set `CUBLAS_WORKSPACE_CONFIG=:16:8` or `CUBLAS_WORKSPACE_CONFIG=:4096:2` (available for CUDNN version 10.2 and above, refer to [CUDNN 8 release note](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-894/release-notes/index.html)).

<a name="4-7"></a>

##### Q3.8 How does the ERNIE model return intermediate layer outputs?

**A:** The current API design does not retain intermediate layer outputs, but it is straightforward to modify the source code in PaddleNLP. Additionally, you can in `ErnieModel`...
```python
def __init__(self):
    # Register a forward_post_hook function for the Layer whose output you want to retain
    self.layer.register_forward_post_hook(self.forward_post_hook)

def forward_post_hook(self, module, input, output):
    # Save the Layer's output to a global List
    global_output_list.append(output)
```

The `forward_post_hook` function will be invoked after the `forward` function call and will save the Layer's output to a global `List`. For more details, refer to [`register_forward_post_hook()`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#register_forward_post_hook).

<a name="deployment-issues"></a>

### Prediction Deployment

<a name="5-1"></a>

##### Q3.9 How to deploy a trained PaddleNLP model to a server?

**A:** We recommend developing in dynamic graph mode and deploying in static graph mode.

1. **Dynamic to Static Conversion**

   Dynamic to static conversion involves transforming a dynamic graph model into a static graph model suitable for deployment. The dynamic graph interface is more user-friendly, offering an interactive Python-style programming experience, which is more conducive to model development. In contrast, static graphs have a significant performance advantage over dynamic graphs. Thus, dynamic to static conversion provides a bridge that balances development cost and performance. You can refer to the official documentation [Dynamic to Static Graph Documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/04_dygraph_to_static/index_cn.html) and use `paddle.jit.to_static` to complete the conversion. Additionally, PaddleNLP provides examples for exporting static graph models.

2. **Deployment with Paddle Inference**

   The model saved after dynamic to static conversion can be deployed with high performance using Paddle Inference. Paddle Inference includes high-performance CPU/GPU Kernels, combined with fine-grained OP horizontal and vertical fusion strategies, and integrates TensorRT to enhance model inference performance. For more details, refer to the documentation [Paddle Inference Introduction](https://paddleinference.paddlepaddle.org.cn/master/product_introduction/inference_intro.html). To help new users understand how to use Paddle Inference with NLP models, PaddleNLP also provides corresponding examples for reference, which can be found in the deploy directory under [/PaddleNLP/examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/).

<a name="5-2"></a>

##### Q3.10 How to convert a static graph model into a dynamic graph model?

**A:** First, you need to save the static graph parameters as `ndarray`.
Data, then map the static graph parameter names to the corresponding dynamic graph parameter names, and finally save them as dynamic graph parameters. For more details, refer to the [parameter conversion script](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie/static_to_dygraph_params).

<a name="NLP Application Scenarios"></a>

### ⭐️Consultation on Specific Models and Application Scenarios

<a name="6-1"></a>

##### Q4.1 【Lexical Analysis】How to customize labels in the LAC model and continue training?

**A:** Update the label file `tag.dict` and modify the number of labels in CRF accordingly.

Refer to the [custom label example](https://github.com/PaddlePaddle/PaddleNLP/issues/662) and the [incremental training custom LABEL example](https://github.com/PaddlePaddle/PaddleNLP/issues/657).

<a name="6-2"></a>

##### Q4.2 In information extraction tasks, is it recommended to use a pre-trained model + CRF, and how can it be implemented?

**A:** A pre-trained model + CRF is a general method for sequence labeling. Currently, pre-trained models have a strong ability to express sequence information, and it is also possible to directly use pre-trained models for sequence labeling tasks.

<a name="6-3"></a>

##### Q4.3 【Reading Comprehension】How to understand `batched=True` in the `map()` method of `MapDatasets`, and why must the parameter `batched` be set to `True` in reading comprehension tasks?

**A:** `batched=True` means performing the map operation on an entire batch (not necessarily the training batch, but a group of data), i.e., the trans_func in the map accepts a group of data as input rather than mapping each item individually. In reading comprehension tasks, depending on the doc_stride used, a single sample may be converted into multiple features, making individual data mapping infeasible, hence the need to set `batched=True`.
。

<a name="6-4"></a>

##### Q4.4 【Semantic Matching】What is the difference between semantic indexing and semantic matching?

**A:** The core issue that semantic indexing aims to solve is how to quickly and accurately find documents related to a query from a massive number of Docs using ANN indexing. Semantic matching, on the other hand, focuses on modeling more detailed semantic matching information between the query and documents. From another perspective, [semantic indexing](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/semantic_indexing) addresses the recall problem in search and recommendation scenarios, whereas [semantic matching](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/text_matching) deals with the ranking problem. Although the problems they solve are different and the solutions they employ vary significantly, there are some common technical points between them that can be mutually referenced.

<a name="6-5"></a>

##### Q4.5 【Word Tagging】How to customize the addition of named entities and corresponding word classes in the wordtag model?

**A:** It primarily relies on reconstructing data for finetuning, while also updating the termtree information. The wordtag process consists of two steps:
(1) Tokenization using the BIOES scheme;
(2) Matching the tokenized information with the TermTree.
    Therefore, we need to:
(1) Ensure correct tokenization, which may depend on the finetune data of wordtag to achieve accurate tokenization;
(2) In wordtag, it is also necessary to annotate the term with the corresponding knowledge information after correct tokenization. The method for customizing the TermTree in wordtag will be provided in future versions.

Refer to [issue](https://github.com/PaddlePaddle/PaddleNLP/issues/822).

<a name="使用咨询问题"></a>

### ⭐️Other Usage Inquiries

<a name="7-1"></a>

##### Q5.1 Error when using PaddleNLP with CUDA11?

**A:** For installation with CUDA11, refer to [issue](https://github.com/PaddlePaddle/PaddleNLP/issues/348). For installation with other CUDA versions, refer to the [official documentation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html).

<a name="7-2"></a>

##### Q5.2 How to set parameters?

**A:** There are multiple methods:
(1) You can set parameters using `set_value()`, where the argument for `set_value()` can be either `numpy` or `tensor`.
```python
   layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie.config["initializer_range"],
                        shape=layer.weight.shape))
```
(2) Set parameters using `create_parameter()`.

``` python
    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            w_tmp = self.create_parameter([1,1])
            self.add_parameter("w_tmp", w_tmp)

        def forward(self, input):
            return self._linear(input)

    mylayer = MyLayer()
    for name, param in mylayer.named_parameters():
        print(name, param)
```

<a name="7-3"></a>

##### Q5.3 Can the GPU version of Paddle run on a CPU, or is a GPU device necessary?

**A:** Devices that do not support GPU can only install the CPU version of PaddlePaddle. If you want the GPU version of PaddlePaddle to run only on the CPU, you can set it by using `export CUDA_VISIBLE_DEVICES=-1`.

<a name="7-4"></a>

##### Q5.4 How to specify whether to train the model using CPU or GPU?

**A:** Generally, our training scripts provide a `--device` option, allowing users to specify the device through `--device`.
Select the device to be used.

Specifically, in a Python file, we can set it to `gpu` or `cpu` using `paddle.device.set_device()`. Refer to the [set_device documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device).

<a name="7-5"></a>

##### Q5.5 Are the prediction results of dynamic graph models and static graph models consistent?

**A:** Under normal circumstances, the prediction results should be consistent. If you encounter inconsistencies, please promptly report them to the PaddleNLP developers for resolution.

<a name="7-6"></a>

##### Q5.6 How can I visualize acc, loss curves, model network structure diagrams, etc.?

**A:** You can use [VisualDL](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/index_cn.html) for visualization. For visualizing acc and loss curves, refer to the [Scalar—Line Chart Component](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl_usage_cn.html#scalar) usage guide. For visualizing model network structures, refer to the [Graph—Network Structure Component](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl_usage_cn.html#graph) usage guide.
