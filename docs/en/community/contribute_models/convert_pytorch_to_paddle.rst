==========================================
Model Format Conversion
==========================================

0. Preface
------------------------------------------
This article explains how to convert model weights between different frameworks (using PyTorch to Paddle framework conversion as an example).

The model format conversion process requires users to have a detailed understanding of the model architecture. Successfully completing the conversion will help deepen users' understanding of the model structure. Let's begin this interesting journey!

1. Overview of Model Weight Files
------------------------------------------
Regardless of the framework, when saving trained models, we need to persist the model's parameter weights. When loading a saved model, we need to load these parameter weights and reassign them to the corresponding model.

Both PyTorch and Paddle frameworks use serialization and deserialization of the model's ``state dict`` (state dictionary) for parameter storage and loading. From a data structure perspective, the ``state dict`` is a dictionary (e.g., Python dict) where keys are model parameter names (string type) and values are corresponding parameters (Tensor type). During saving, the framework first retrieves the target object's ``state dict`` and saves it to disk. During loading, the framework first loads the saved ``state dict`` from disk and applies it to the target object using the ``set_state_dict()`` method.

By convention, model files saved in Paddle framework typically use the `.pdparams` suffix, while PyTorch models commonly use `.pt`, `.pth`, or `.bin` suffixes. Although the suffix doesn't affect model saving/loading, we generally follow this naming convention.

2. Overview of Model ``state dict``
------------------------------------------
Now that we've briefly introduced model files and their stored ``state dict``, let's examine a concrete example to better understand the ``state dict``.

``LeNet``, proposed by Yann LeCun et al. in 1998, is a CNN model successfully applied to handwritten digit recognition. Paddle provides an integrated implementation of ``LeNet``. The following code demonstrates loading this model and outputting its corresponding ``state dict``:

.. code:: python
    from paddle.vision.models import LeNet

    # Instantiate model
    model = LeNet()
    
    # Print state dict keys
    print("State dict keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # Print state dict contents
    print("\nState dict items:")
    for key, value in model.state_dict().items():
        print(f"{key}: {value.shape}")

The output shows the model's parameter names and shapes:

```
State dict keys:
conv1.weight
conv1.bias
conv2.weight
conv2.bias
fc1.weight
fc1.bias
fc2.weight
fc2.bias
fc3.weight
fc3.bias

State dict items:
conv1.weight: [6, 1, 5, 5]
conv1.bias: [6]
conv2.weight: [16, 6, 5, 5]
conv2.bias: [16]
fc1.weight: [120, 400]
fc1.bias: [120]
fc2.weight: [84, 120]
fc2.bias: [84]
fc3.weight: [10, 84]
fc3.bias: [10]
```

This output reveals several key characteristics of the ``state dict``:
1. Parameter names follow strict naming conventions reflecting the model's architecture
2. Parameter shapes correspond to layer configurations
3. Contains both weight and bias parameters for each layer
4. Parameter names directly reflect their position in the network

This example demonstrates how the ``state dict`` faithfully records all model parameters and their hierarchical relationships.
python

    >>> import paddle
    >>> from paddle.vision.models import LeNet
    >>> model = LeNet()
    >>> model.state_dict().keys()  # Output all keys of state_dict
    odict_keys(['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias',
                'fc.0.weight', 'fc.0.bias', 'fc.1.weight', 'fc.1.bias', 'fc.2.weight', 'fc.2.bias'])

    >>> model.state_dict()['features.0.weight']  # Output the value corresponding to 'features.0.weight'
    Parameter containing:
    Tensor(shape=[6, 1, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[[[-0.31584871,  0.27280194, -0.43816274],
              [ 0.06681869,  0.44526964,  0.80944657],
              [ 0.05796078,  0.57411081,  0.15335406]]],
            ...
            ...
            [[[-0.07211500, -0.14458601, -1.11733580],
              [ 0.53036308, -0.19761689,  0.56962037],
              [-0.09760553, -0.02011104, -0.50577533]]]])

We can obtain all parameter names of the model through ``model.state_dict().keys()``. 
The ``LeNet`` model contains 10 parameter groups: *'features.0.weight'*, *'features.0.bias'*, *'features.3.weight'*,
*'features.3.bias'*, *'fc.0.weight'*, *'fc.0.bias'*, *'fc.1.weight'*, *'fc.1.bias'*, *'fc.2.weight'*, and *'fc.2.bias'*.

By querying ``model.state_dict()['features.0.weight']``, we can examine the specific weight values of the **'features.0.weight'** parameter.
The output shows this weight is a Tensor with dtype=float32 and shape=[6, 1, 3, 3].

3. Weight Format Conversion Using ``state_dict``
------------------------------------------------
After understanding model storage/loading and related ``state_dict``, let's examine the concrete steps for model format conversion.
Generally, we can perform model format conversion through ``state_dict``.
The mutual conversion of **state dict** can assist us in model format conversion.

Taking model weight conversion from PyTorch to Paddle framework as an example, the specific conversion process is:

1. Load PyTorch model to obtain ``state dict``
2. Convert PyTorch's ``state dict`` to Paddle's ``state dict``
3. Save Paddle's ``state dict`` to obtain Paddle model.

Let's examine a concrete example: ``'bert-base-uncased'`` is a 12-layer BERT English model open-sourced by Google. This model is integrated in both PaddleNLP (Paddle framework) and HuggingFace's transformers (PyTorch framework), with completely identical parameter quantities and specific parameter values. We can compare the ``state dict`` of these two models to understand conversion details.

3.1 ``state dict`` in PyTorch Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, load the ``'bert-base-uncased'`` model from transformers:

.. code::
>>> import torch
>>> model_name = "bert-base-uncased"
>>> # Model download: https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin
>>> model_file = "pytorch_model.bin"
>>> pytorch_state_dict = torch.load(model_file)
>>> pytorch_state_dict.keys()
odict_keys(['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight',
            'bert.embeddings.LayerNorm.gamma', 'bert.embeddings.LayerNorm.beta',
            'bert.encoder.layer.0.attention.self.query.weight', 'bert.encoder.layer.0.attention.self.query.bias',
            'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.key.bias',
            'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.value.bias',
            'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.output.dense.bias',
            'bert.encoder.layer.0.attention.output.LayerNorm.gamma', 'bert.encoder.layer.0.attention.output.LayerNorm.beta',
            'bert.encoder.layer.0.intermediate.dense.weight', 'bert.encoder.layer.0.intermediate.dense.bias',
            'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.output.dense.bias',
            'bert.encoder.layer.0.output.LayerNorm.gamma', 'bert.encoder.layer.0.output.LayerNorm.beta',
            'bert.encoder.layer.1'...
            'bert.encoder.layer.2'...
            .
            .
            .
            'bert.encoder.layer.9'...
            'bert.encoder.layer.10'...
            'bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias',
            'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias',
            'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias',
            'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias',
            'bert.encoder.layer.11.attention.output.LayerNorm.gamma', 'bert.encoder.layer.11.attention.output.LayerNorm.beta',
            'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias',
            'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias',
            'bert.encoder.layer.11.output.LayerNorm.gamma', 'bert.encoder.layer.11.output.LayerNorm.beta',
            'bert.pooler.dense.weight', 'bert.pooler.dense.bias',
            'cls.predictions.bias', 'cls.predictions.transform.dense.weight',
            'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.gamma',
            'cls.predictions.transform.LayerNorm.beta', 'cls.predictions.decoder.weight',
            'cls.seq_relationship.weight', 'cls.seq_relationship.bias'])

**odict_keys** (ordered dictionary keys) displays the ordered dictionary keys corresponding to the PyTorch model file
```state dict``` keys:
Upon closer inspection, we can categorize the parameters into several major modules: **embeddings** module, **encoder_layers** module, **pooler** module, and **cls** module.

Let's interpret each module in conjunction with BERT's actual structure:

- **embeddings** module

  Parameters starting with *'bert.embeddings'* belong to the embeddings module, including the word_embeddings matrix, position_embeddings matrix, token_type_embeddings matrix, and LayerNorm layer parameters in the embeddings module.
  
- **encoder_layers** module

  Parameters starting with *'bert.encoder.layer'* correspond to each encoder layer. We observe that the ```'bert-base-uncased'``` model contains 12 encoder layers (numbered 0-11), all sharing identical architecture. Each encoder layer mainly consists of a *self-attention* module and a *feed-forward* module. Let's examine the parameters of the first encoder layer (numbered 0, parameters starting with 'bert.encoder.layer.0'):

  First, the *self-attention* module:

  * *'attention.self.query'*, *'attention.self.key'*, and *'attention.self.value'* represent the query matrix, key matrix, and value matrix in the self-attention structure.
  * *'attention.output.dense'* is the linear layer in the self-attention structure.
  * *'attention.output.LayerNorm'* denotes the LayerNorm layer following the self-attention structure.

  Next, the *feed-forward* module corresponds to parameters starting with 'intermediate.dense' and 'output.dense'. After the feed-forward layer, there's another *LayerNorm* layer, corresponding to parameters starting with 'output.LayerNorm'.

- **pooler** module

  The pooler module follows the last encoder layer, performing pooling operations on the output of the final encoder layer.

- **cls** module

  The cls module calculates parameters for MLM (masked language model) and NSP (next sentence prediction) tasks. Parameters starting with 'cls.predictions' are for MLM tasks, while 'cls.seq_relationship' parameters are for NSP prediction tasks.

3.2 ```state dict``` in Paddle Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We should now have a deeper understanding of BERT's architecture and corresponding parameters. Next, let's load the model from PaddleNLP:

.. code::
>>> import paddle
>>> model_name = "bert-base-uncased"
>>> # Model download URL: https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams
>>> model_file = "bert-base-uncased.pdparams"
>>> paddle_state_dict = paddle.load(model_file)
>>> paddle_state_dict.keys()
dict_keys(['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight',
            'bert.embeddings.layer_norm.weight', 'bert.embeddings.layer_norm.bias',
            'bert.encoder.layers.0.self_attn.q_proj.weight', 'bert.encoder.layers.0.self_attn.q_proj.bias',
            'bert.encoder.layers.0.self_attn.k_proj.weight', 'bert.encoder.layers.0.self_attn.k_proj.bias',
            'bert.encoder.layers.0.self_attn.v_proj.weight', 'bert.encoder.layers.0.self_attn.v_proj.bias',
            'bert.encoder.layers.0.self_attn.out_proj.weight', 'bert.encoder.layers.0.self_attn.out_proj.bias',
            'bert.encoder.layers.0.linear1.weight', 'bert.encoder.layers.0.linear1.bias',
            'bert.encoder.layers.0.linear2.weight', 'bert.encoder.layers.0.linear2.bias',
            'bert.encoder.layers.0.norm1.weight', 'bert.encoder.layers.0.norm1.bias',
            'bert.encoder.layers.0.norm2.weight', 'bert.encoder.layers.0.norm2.bias',
            'bert.encoder.layers.1'...
            ...
            ...
            'bert.encoder.layers.10'...
            'bert.encoder.layers.11.self_attn.q_proj.weight', 'bert.encoder.layers.11.self_attn.q_proj.bias',
            'bert.encoder.layers.11.self_attn.k_proj.weight', 'bert.encoder.layers.11.self_attn.k_proj.bias',
            'bert.encoder.layers.11.self_attn.v_proj.weight', 'bert.encoder.layers.11.self_attn.v_proj.bias',
            'bert.encoder.layers.11.self_attn.out_proj.weight', 'bert.encoder.layers.11.self_attn.out_proj.bias',
            'bert.encoder.layers.11.linear1.weight', 'bert.encoder.layers.11.linear1.bias',
            'bert.encoder.layers.11.linear2.weight', 'bert.encoder.layers.11.linear2.bias',
            'bert.encoder.layers.11.norm1.weight', 'bert.encoder.layers.11.norm1.bias',
            'bert.encoder.layers.11.norm2.weight', 'bert.encoder.layers.11.norm2.bias',
            'bert.pooler.dense.weight', 'bert.pooler.dense.bias',
            'cls.predictions.decoder_weight', 'cls.predictions.decoder_bias',
            'cls.predictions.transform.weight', 'cls.predictions.transform.bias',
            'cls.predictions.layer_norm.weight', 'cls.predictions.layer_norm.bias',
            'cls.seq_relationship.weight', 'cls.seq_relationship.bias'])
``` 
The `state dict` is stored using a dict. We can see that the `state dict` of both frameworks are highly similar. Let's compare them:

- The storage mechanisms are similar: PyTorch uses an ordered_dict from Python to store model parameter states, while Paddle uses a standard Python dict.
- The structures are also similar: both can be divided into embeddings, encoder_layer, pooler, cls, etc. (This is intuitive since the model architectures and parameters are identical).
- However, there are some differences: there are subtle variations in the keys of the `state dict` between the two, which result from differences in parameter naming conventions in their respective implementations.

3.3 Comparison of PyTorch and Paddle `state dict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Next, we will perform a detailed parameter name and weight mapping between the two `state dict`s. The following table presents the organized `state_dict` comparison:
```
Correspondence Table (Parameters in the same row are corresponding):

+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| Keys (PyTorch)                                         | Shape (PyTorch)            | Keys (Paddle)                                    | Shape (Paddle)            |
+========================================================+============================+==================================================+===========================+
| bert.embeddings.word_embeddings.weight                 | [30522, 768]               | bert.embeddings.word_embeddings.weight           | [30522, 768]              |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.position_embeddings.weight             | [512, 768]                 | bert.embeddings.position_embeddings.weight       | [512, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.token_type_embeddings.weight           | [2, 768]                   | bert.embeddings.token_type_embeddings.weight     | [2, 768]                  |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.LayerNorm.gamma                        | [768]                      | bert.embeddings.layer_norm.weight                | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.embeddings.LayerNorm.beta                         | [768]                      | bert.embeddings.layer_norm.bias                  | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.query.weight       | [768, 768]                 | bert.encoder.layers.0.self_attn.q_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.query.bias         | [768]                      | bert.encoder.layers.0.self_attn.q_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.key.weight         | [768, 768]                 | bert.encoder.layers.0.self_attn.k_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.key.bias           | [768]                      | bert.encoder.layers.0.self_attn.k_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.value.weight       | [768, 768]                 | bert.encoder.layers.0.self_attn.v_proj.weight    | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.self.value.bias         | [768]                      | bert.encoder.layers.0.self_attn.v_proj.bias      | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.dense.weight     | [768, 768]                 | bert.encoder.layers.0.self_attn.out_proj.weight  | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.dense.bias       | [768]                      | bert.encoder.layers.0.self_attn.out_proj.bias    | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.LayerNorm.gamma  | [768]                      | bert.encoder.layers.0.norm1.weight               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.attention.output.LayerNorm.beta   | [768]                      | bert.encoder.layers.0.norm1.bias                 | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.intermediate.dense.weight         | [3072, 768]                | bert.encoder.layers.0.linear1.weight             | [768, 3072]               |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.intermediate.dense.bias           | [3072]                     | bert.encoder.layers.0.linear1.bias               | [3072]                    |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.dense.weight               | [768, 3072]                | bert.encoder.layers.0.linear2.weight             | [3072, 768]               |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.dense.bias                 | [768]                      | bert.encoder.layers.0.linear2.bias               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.LayerNorm.gamma            | [768]                      | bert.encoder.layers.0.norm2.weight               | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.encoder.layer.0.output.LayerNorm.beta             | [768]                      | bert.encoder.layers.0.norm2.bias                 | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.pooler.dense.weight                               | [768, 768]                 | bert.pooler.dense.weight                         | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| bert.pooler.dense.bias                                 | [768]                      | bert.pooler.dense.bias                           | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.bias                                   | [30522]                    | cls.predictions.decoder_bias                     | [30522]                   |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.dense.weight                 | [768, 768]                 | cls.predictions.transform.weight                 | [768, 768]                |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.dense.bias                   | [768]                      | cls.predictions.transform.bias                   | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.LayerNorm.gamma              | [768]                      | cls.predictions.layer_norm.weight                | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.transform.LayerNorm.beta               | [768]                      | cls.predictions.layer_norm.bias                  | [768]                     |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.predictions.decoder.weight                         | [30522, 768]               | cls.predictions.decoder_weight                   | [30522, 768]              |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.seq_relationship.weight                            | [2, 768]                   | cls.seq_relationship.weight                      | [768, 2]                  |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
| cls.seq_relationship.bias                              | [2]                        | cls.seq_relationship.bias                        | [2]                       |
+--------------------------------------------------------+----------------------------+--------------------------------------------------+---------------------------+
The parameters and weights of the ``state dict`` help us correctly perform the conversion of the ``state dict``.

From the parameter names, we can observe a basic correspondence, for example:

* `bert.embeddings.LayerNorm.gamma` corresponds to `bert.embeddings.layer_norm.weight`;
* `bert.embeddings.LayerNorm.beta` corresponds to `bert.embeddings.layer_norm.bias`;
* `bert.encoder.layer.0.attention.self.query.weight` corresponds to `bert.encoder.layers.0.self_attn.q_proj.weight`;
* `bert.encoder.layer.0.attention.self.query.bias` corresponds to `bert.encoder.layers.0.self_attn.q_proj.bias`.

The order of parameters is generally consistent, but there are exceptions, such as:

* `bert.encoder.layers.0.norm1.weight` corresponds to `bert.encoder.layer.0.attention.output.LayerNorm.gamma`;
* `bert.encoder.layers.0.norm1.bias` corresponds to `bert.encoder.layer.0.attention.output.LayerNorm.beta`;
* `bert.encoder.layer.0.intermediate.dense.weight` corresponds to `bert.encoder.layers.0.linear1.weight`;
* `bert.encoder.layer.0.output.dense.weight` corresponds to `bert.encoder.layers.0.linear2.weight`;
* `bert.encoder.layer.0.output.LayerNorm.gamma` corresponds to `bert.encoder.layers.0.norm2.weight`.

The correct parameter correspondence may require us to examine the specific code for determination. We have already established accurate one-to-one mapping of the keys in the table above. After establishing the key correspondences, we can proceed with value mapping.

If you observe carefully, you'll notice that some parameter values have different shapes. For example, the corresponding parameters `bert.encoder.layer.0.intermediate.dense.weight` and `bert.encoder.layers.0.linear1.weight` have value shapes of `[3072, 768]` and `[768, 3072]` respectively, which are transposes of each other. This is because PyTorch saves the weights of `nn.Linear` modules in transposed form. Therefore, when processing the `state dict`, we need to perform corresponding transpose operations on these linear layer parameters.
When converting, it is necessary to properly handle the shape conversion (for example, transposing the parameter weights of the nn.Linear layer in a PyTorch model and then generating the corresponding parameter weights for Paddle).

Additional details to note include several potential scenarios:

- Some model structures may have parameter processing differences leading to parameter splitting/merging operations. In such cases, we need to perform many-to-one or one-to-many parameter mappings while splitting/merging corresponding values.
- When batch norm layers are present, we need to pay attention to the todo.

3.4 bert model conversion code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is the most critical model conversion phase. This step is crucial as only correct conversion of the ``state dict`` can ensure successful accuracy validation.

Below is the model conversion code (PyTorch to Paddle):

.. code:: python

    # Model conversion core code
    def convert_bert_model(pretrained_torch_model, paddle_model):
        torch_state_dict = pretrained_torch_model.state_dict()
        paddle_state_dict = {}
        
        # Embedding layer conversion
        paddle_state_dict["embeddings.word_embeddings.weight"] = torch_state_dict["bert.embeddings.word_embeddings.weight"].detach().numpy()
        # Add other embedding parameters...
        
        # Encoder layers conversion
        for i in range(config.num_hidden_layers):
            # Attention weights
            paddle_state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = torch_state_dict[f"bert.encoder.layer.{i}.attention.self.query.weight"].T.numpy()
            # Add bias and other linear transformations...
            
            # LayerNorm parameters
            paddle_state_dict[f"encoder.layers.{i}.norm1.weight"] = torch_state_dict[f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy()
            # Continue processing all parameters...
        
        # Load converted parameters into Paddle model
        paddle_model.set_dict(paddle_state_dict)
        return paddle_model
```python

    import paddle
    import torch
    import numpy as np

    torch_model_path = "pytorch_model.bin"
    torch_state_dict = torch.load(torch_model_path)

    paddle_model_path = "bert_base_uncased.pdparams"
    paddle_state_dict = {}

    # State_dict's keys mapping: from torch to paddle
    keys_dict = {
        # about embeddings
        "embeddings.LayerNorm.gamma": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.beta": "embeddings.layer_norm.bias",

        # about encoder layer
        'encoder.layer': 'encoder.layers',
        'attention.self.query': 'self_attn.q_proj',
        'attention.self.key': 'self_attn.k_proj',
        'attention.self.value': 'self_attn.v_proj',
        'attention.output.dense': 'self_attn.out_proj',
        'attention.output.LayerNorm.gamma': 'norm1.weight',
        'attention.output.LayerNorm.beta': 'norm1.bias',
        'intermediate.dense': 'linear1',
        'output.dense': 'linear2',
        'output.LayerNorm.gamma': 'norm2.weight',
        'output.LayerNorm.beta': 'norm2.bias',

        # about cls predictions
        'cls.predictions.transform.dense': 'cls.predictions.transform',
        'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
        'cls.predictions.transform.LayerNorm.gamma': 'cls.predictions.layer_norm.weight',
        'cls.predictions.transform.LayerNorm.beta': 'cls.predictions.layer_norm.bias',
        'cls.predictions.bias': 'cls.predictions.decoder_bias'
    }


    for torch_key in torch_state_dict:
        paddle_key = torch_key
        for k in keys_dict:
            if k in paddle_key:
                paddle_key = paddle_key.replace(k, keys_dict[k])

        if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

        print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
        print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

    paddle.save(paddle_state_dict, paddle_model_path)


Let's take a look at this conversion code:
We need to download the PyTorch model to be converted, load the model to obtain **torch_state_dict**; **paddle_state_dict** and **paddle_model_path** define the converted
``state dict`` and model file paths;
The **keys_dict** in the code defines the key mappings between them (can be verified by comparing with the table above).

The next step is the most crucial *paddle_state_dict* construction. We map each key in the *torch_state_dict*,
to obtain the corresponding key in the *paddle_state_dict*. After obtaining the *paddle_state_dict* key, we need
to convert the value from *torch_state_dict*. If the corresponding structure is an ``nn.Linear`` module,
we also need to perform transpose operation on the value.

Finally, by saving the obtained *paddle_state_dict*, we get the corresponding Paddle model.
Thus, we have completed the model conversion and obtained the Paddle framework model ``"model_state.pdparams"``.

4. Model Weight Validation
------------------------------------------
After obtaining the model weights, we need to verify the conversion correctness through precision alignment.
We can validate through forward inference and downstream task fine-tuning.

4.1 Forward Precision Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Forward precision alignment is straightforward. We only need to ensure consistent input for both frameworks,
and verify whether the outputs match. Some important notes: we need to run inference in eval mode,
set dropout to 0, and other operations to eliminate randomness effects.

In addition to the model weight file, we need to prepare the model configuration file. By placing both the model weights (model_state.pdparams)
and the model configuration file (model_config.json) in the same directory, we can perform forward precision validation.
The following provides a code example for BERT model forward alignment:

.. code::
```python
text = "Welcome to use paddle paddle and paddlenlp!"
torch_model_name = "bert-base-uncased"
paddle_model_name = "bert-base-uncased"

# torch output
import torch
import transformers
from transformers.models.bert import *

# torch_model = BertForPreTraining.from_pretrained(torch_model_name)
torch_model = BertModel.from_pretrained(torch_model_name)
torch_tokenizer = BertTokenizer.from_pretrained(torch_model_name)
torch_model.eval()

torch_inputs = torch_tokenizer(text, return_tensors="pt")
torch_outputs = torch_model(**torch_inputs)

torch_logits = torch_outputs[0]
torch_array = torch_logits.cpu().detach().numpy()
print("torch_prediction_logits shape:{}".format(torch_array.shape))
print("torch_prediction_logits:{}".format(torch_array))


# paddle output
import paddle
import paddlenlp
from paddlenlp.transformers.bert.modeling import *
import numpy as np

# paddle_model = BertForPretraining.from_pretrained(paddle_model_name)
paddle_model = BertModel.from_pretrained(paddle_model_name)
paddle_tokenizer = BertTokenizer.from_pretrained(paddle_model_name)
paddle_model.eval()

paddle_inputs = paddle_tokenizer(text)
paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
paddle_outputs = paddle_model(**paddle_inputs)

paddle_logits = paddle_outputs[0]
paddle_array = paddle_logits.numpy()
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))


# the output logits should have the same shape
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))

The code will finally print the maximum difference between corresponding elements in the output matrices, which can be used to verify if we have aligned the forward accuracy.

4.2 Downstream Task Fine-tuning Verification (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the forward accuracy is aligned, it generally indicates successful model conversion. We can also run downstream task fine-tuning for double-checking.
Similarly, we need to use identical training data, same training parameters, and same training environment to compare the convergence behavior and metrics.

5. Final Remarks
------------------------------------------
Congratulations on successfully completing the model weight format conversion work! We welcome you to contribute your models to PaddleNLP via PR,
so that every PaddleNLP user can benefit from your shared models!