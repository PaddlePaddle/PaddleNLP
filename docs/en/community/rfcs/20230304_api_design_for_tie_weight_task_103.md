# No.103: Add tie_weights Capability

| API Name     | New API Name                                      |
|-------------|--------------------------------------------------|
| Submitter   | Qiu Wenbo, Liu Wangwang                          |
| Submit Date | 2023-03-10                                      |
| Version     | V3                                              |
| PaddlePaddle Version | Should be based on develop version unless otherwise specified |
| File Name   | 20230304_api_design_for_tie_weight_task_103.md   |

# 1. Overview
## 1.1 Background
Corresponding to task No.103: Add tie_weights capability

Weight tying (tie_weights) typically refers to sharing weights between input embeddings and output embeddings, thereby reducing network parameters and allowing more sufficient training of embedding layer parameters.

This technique is mentioned in "Attention Is All You Need" (Transformer paper), specifically in section 3.4, which describes sharing weights between encoder input embeddings, decoder input embeddings, and output linear layers. The effectiveness of this technique is validated in the paper "Using the Output Embedding to Improve Language Models".

Therefore, pre-trained language models need to implement a weight sharing function between input embeddings and output embeddings for user convenience.

Related issues:
* [https://github.com/PaddlePaddle/PaddleNLP/issues/4740](https://github.com/PaddlePaddle/PaddleNLP/issues/4740)

## 1.2 Objectives
Add a fundamental function to pre-trained language models to implement weight sharing between input embeddings and output embeddings:

- Add tie_weights functionality to PaddleNLP, aligning with HuggingFace Transformers' [tie_weights](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.tie_weights) implementation
- Reference: [https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172)

## 1.3 Significance
Implementing the weight tying function as a model technique to improve training efficiency and reduce model parameters.

As a fundamental model function, it facilitates experimental setup during pre-trained model architecture design, reducing model parameters and enhancing model performance.

# 2. Current Status of PaddlePaddle
Research on current framework support for this functionality. If not supported, investigate possible alternatives:

PaddlePaddle currently lacks a unified implementation for weight tying. Users need to implement this manually.

Some example code in PaddleNLP shows implementations:

(1) [Code Link 1](https://github.com/qiuwenbogdut/PaddleNLP/blob/develop/examples/language_model/transformer-xl/mem_transformer.py#L811)
```python
if tie_weight:
        for i in range(len(self.crit.out_layers_weight)):
            self.crit.out_layers_weight[i] = self.word_emb.emb_layers[i].weight

if tie_projs:
        for i, tie_proj in enumerate(tie_projs):
            if tie_proj and div_val == 1 and d_model != d_embed:
                self.crit.out_projs[i] = self.word_emb.emb_projs[0]
            elif tie_proj and div_val != 1:
                self.crit.out_projs[i] = self.word_emb.emb_projs[i]
```

(2) [Code Link 2](https://github.com/PaddlePaddle/PaddleNLP/blob/4e5df921ff61ddae1d869c37aea621b9cac6bcd4/paddlenlp/transformers/reformer/modeling.py#L1977)

```python
def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        tie_word_embeddings = (
            self.tie_word_embeddings
            if hasattr(self, "tie_word_embeddings")
            else self.config.get("tie_word_embeddings", False)
        )
        if hasattr(self, "get_output_embeddings") and hasattr(self, "get_input_embeddings") and tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
```

(3) [Code Link 3](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py#L748)
```python
class ErnieLMPredictionHead(nn.Layer):
    r"""
    Ernie Model with a `language modeling` head on top.
    """

    def __init__(
        self,
        config: ErnieConfig,
        embedding_weights=None,
        weight_attr=None,
    ):
        super(ErnieLMPredictionHead, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size, weight_attr=weight_attr)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder_weight = (
            self.create_parameter(
                shape=[config.vocab_size, config.hidden_size],
                dtype=self.transform.weight.dtype,
                attr=weight_attr,
                is_bias=False,
            )
            if embedding_weights is None
            else embedding_weights
        )
        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
        )
```
In fact, most `tie_weights` implementations in PaddleNLP are directly handled at the model layer definition level, as seen in this [code](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py#L748), rather than being uniformly implemented outside the model like in Transformers. The goal of this project is to explore whether we can achieve this binding externally without requiring per-model implementations.

There are two main approaches to `tie_weights` implementation in Paddle:
* One defines the `tie_weights` function in modeling.py, where the model implements `get_input_embeding()` and `get_output_embeding()` to retrieve input and output embedding layer weights, then binds them via assignment. As shown in code links (1)(2) above.
* The other directly assigns the input embedding's weight to the output layer's weight during model layer definition. The embedding's weight is directly passed to the head to construct the linear output layer, expecting to obtain weights through `get_input_embeding()` and pass them to the head layer, as shown in code link (3) above.

The ideal implementation would be to uniformly handle `tie_weights` in the [base class model_utils.py#L897](https://github.com/PaddlePaddle/PaddleNLP/blob/be80a3e30fb681e53773c265babe611d4df62ead/paddlenlp/transformers/model_utils.py#L897) to reduce developer overhead.

# III. Industry Solution Research
Describe how industry deep learning frameworks implement this feature, including current status and future trends. Research scope includes but not limited to TensorFlow, PyTorch, NumPy, etc.

(1) Currently, the `tie_weights` function is implemented in Hugging Face's Transformers library. [Code link](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172)
```python
def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
```

(2) Tensor2Tensor library's tie_weights implementation code [code link](https://github.com/tensorflow/tensor2tensor/blob/316c9ce2f2b2373f44f5be0da712dda3e5861a75/tensor2tensor/layers/modalities.py#L1106)
```python
def symbol_top(body_output, targets, model_hparams, vocab_size):
  del targets  # unused arg
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False
  with tf.variable_scope(scope_name, reuse=reuse):
    body_output_shape = common_layers.shape_list(body_output)
    var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
    if (model_hparams.factored_logits and
        model_hparams.mode == tf_estimator.ModeKeys.TRAIN):
      # insert channels dimension
      body_output = tf.expand_dims(body_output, 3)
      return common_layers.FactoredTensor(body_output, var)
    else:
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      logits = tf.matmul(body_output, var, transpose_b=True)
      return tf.reshape(logits,
                        body_output_shape[:-1] + [1, vocab_size])
```

(3) Implementation of weight tying in fairseq library [code link](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/fconv.py#L480)
```python
self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, (
                    "Shared embedding weights require matching dimensions "
                    "out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                )
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)
```
# 4. Comparative Analysis
Both Paddle and HuggingFace's Transformers are developed based on dynamic graphs. Therefore, we plan to implement the functionality by referencing the implementation approach of the tie_weight function from HuggingFace's Transformers.

# 5. Design Approach and Implementation Plan
We will implement the solution based on Paddle by referencing the implementation approach from HuggingFace's Transformers.

Implementation steps for the tie_weight function:
1. Obtain the weight object A of the model's input embedding
2. Obtain the weight object B of the model's output embedding
3. Make both A and B point to the same weight value

## Naming and Parameter Design
Reference: [PaddlePaddle API Design and Naming Conventions](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)

## Underlying OP Design

## API Implementation Plan

# 6. Testing and Validation Considerations
Reference: [New API Testing and Validation Standards](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

Two methods to test tie_weight:
* Directly check the id consistency between the output layer weight and input layer weight. If consistent, pass; otherwise fail.
* Train for several steps, after several backward passes, check if the output layer weight and input layer weight remain consistent. If consistent, pass; otherwise fail.

We will use id consistency check for unit testing: build unit tests to verify whether the id of the weight obtained from get_input_embedding is consistent with the id from get_output_embedding. If consistent, pass; otherwise fail.

# 7. Feasibility Analysis and Timeline Planning

Design a small script to validate this approach:
```python
# Here is a simple validation script example
import paddle

class TestModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(1000, 256)
        self.dense = paddle.nn.Linear(256, 1000)

    def tie_weights(self):
        self.dense.weight = self.embedding.weight

model = TestModel()
model.tie_weights()

print(id(model.embedding.weight) == id(model.dense.weight))  # Should output True
```
```python
import numpy as np
from paddle.nn import Embedding

"""step1 Define two distinct embedding objects AA and BB"""
print('------------step1')
AA = Embedding(1,2)
BB = Embedding(1,2)

AA.weight = BB.weight # Bind the weights

""" step2 Test the binding result"""
print('------------step2')
print('Check if AA and BB have same id:', AA is BB,id(AA), id(BB))
print('Check if AA.weight and BB.weight have same id:',AA.weight is BB.weight,id(AA.weight), id(BB.weight))

print("AA.weight: ",AA.weight)
print("BB.weight: ",BB.weight)



""" step3 Attempt to modify AA's weight value and verify if BB's weight changes accordingly"""
print('------------step3')
AA.weight.set_value(np.array([[4.0,6.0]],dtype=np.float32))

print('Check if modified AA.weight and BB.weight have same id:',AA.weight is BB.weight,id(AA.weight), id(BB.weight))
print("AA.weight after modification: ",AA.weight)
print("BB.weight:",BB.weight)

```

Time and development schedule planning, key milestones
- 3.10 Finalize implementation approach with official team
- 3.17 Submit implementation code

# VIII. Impact Analysis
Open questions requiring further discussion, controversial issues; potential impacts on other modules

# Glossary

# Attachments and References
