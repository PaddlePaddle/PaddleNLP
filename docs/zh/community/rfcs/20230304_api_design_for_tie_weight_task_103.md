# No.103：新增 tie_weights 能力

| API 名称      | 新增 API 名称                                        |
|--------------|----------------------------------------------------|
| 提交作者     | 丘文波, 刘旺旺                                     |
| 提交时间     | 2022-03-10                                         |
| 版本号       | V3                                                 |
| 依赖飞桨版本 | 如无特殊情况，都应基于 develop 版本开发              |
| 文件名       | 20230304_api_design_for_tie_weight_task_103.md<br> |


# 一、概述
## 1、相关背景
对应任务是 No.103：新增 tie_weights 能力

权重绑定, 一般是指将输入层 embedding 和 输出层 embeding 共享权重, 从而在减少网络的参数量, 使得 embeding 层参数训练更加充分.

其中《attention is all you need》中的提到的 transformer 模型也使用到了 tie weigh 这个技巧, 论文3.4节提到将 encoder 输入 embedding 与 decoder 输入 embedding 以及输出线性层权重共享 这个技巧的有效性在论文《Using the output embedding to improve language models》进行了验证 .

所以预训练语言模型需要实现一个输入层 embedding 和 输出层 embeding 共享权重共享功能,方便使用者进行调用.

相关 issue:
* [https://github.com/PaddlePaddle/PaddleNLP/issues/4740](https://github.com/PaddlePaddle/PaddleNLP/issues/4740)


## 2、功能目标
给预训练语言模型增加一个基础函数, 实现输入层 embeding 和输出层 embedding 的权重共享绑定:

- 为 PaddleNLP 新增 tie_weights 功能，能够对齐 HuggingFace Transformers 中的[tie_weights](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.tie_weights)功能
- 参考: [https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172)


## 3、意义
实现权重绑定的函数, 作为一种模型技巧来提升训练效果.减少模型参数,

权重绑定的函数作为模型的一个基本函数, 在基于预训练模型组网的时候 方便进行调用进行实验, 减少模型参数,提升模型效果.


# 二、飞桨现状
对飞桨框架目前支持此功能的现状调研，如果不支持此功能，如是否可以有替代实现的 API，是否有其他可绕过的方式，或者用其他 API 组合实现的方式；

paddle 中并没有对 tie weight 的统一实现,调用者需自己写代码实现这部分功能.

paddleNLP 中的一些示例代码中也找到了一个 tie weight 的实现.

(1) [代码链接1](https://github.com/qiuwenbogdut/PaddleNLP/blob/develop/examples/language_model/transformer-xl/mem_transformer.py#L811)

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

(2) [代码链接2](https://github.com/PaddlePaddle/PaddleNLP/blob/4e5df921ff61ddae1d869c37aea621b9cac6bcd4/paddlenlp/transformers/reformer/modeling.py#L1977)

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

(3) [代码链接3](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py#L748)
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


其实 paddlenlp 内大部分的 tie_weights 实现是直接在模型 layer 定义层面实现的，见[代码](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py#L748)
，而不是类似 transformers 一样在模型以外统一实现的。这个项目的目标就是看一下能否在模型外统一实现，而不用每个模型都自己实现一次

paddle 里面 tie_weghts 实现主要有两种方式:
* 一种在 modeling.py 中定义了 tie_weghts 函数，相应的模型也实现了 get_input_embeding()和 get_output_embeding()来获取输入和输出 embeding 层权重,然后通过赋值方式进行绑定。如上面的代码链接(1)(2)
* 另外一种是 在定义模型层的时候 直接将输入 input_embeding 的 weight，赋值给输出层 weight. 将 embedding 的 weight 直接传给 head 来构建 linear 输出层，期望是在 get_input_embeding()拿到 weight，然后传给 head 层，如上面代码链接(3)



最好是在模型[基类里面 model_utils.py#L897](https://github.com/PaddlePaddle/PaddleNLP/blob/be80a3e30fb681e53773c265babe611d4df62ead/paddlenlp/transformers/model_utils.py#L897)
去统一实现 tie_weights,减少调用者的开发.

# 三、业内方案调研
描述业内深度学习框架如何实现此功能，包括与此功能相关的现状、未来趋势；调研的范围包括不限于 TensorFlow、PyTorch、NumPy 等

(1)目前 huggingface 的 transformers 库中实现了这个 tieweight 这个基础函数. [代码链接](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/modeling_utils.py#L1172)
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


(2) tensor2tensor 库 tieweight 实现代码 [代码链接](https://github.com/tensorflow/tensor2tensor/blob/316c9ce2f2b2373f44f5be0da712dda3e5861a75/tensor2tensor/layers/modalities.py#L1106)
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


(3) fairseq 库 中 tie weight 实现函数 [代码链接](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/fconv.py#L480)
```python
self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, (
                    "Shared embed weights implies same dimensions "
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                )
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)
```

# 四、对比分析
paddle 和 huggingface 的 transformers 都是基于动态图进行开发, 所以准备参照 huggingface 的 transformers  的 tie weight 函数思路去实现功能.

# 五、设计思路与实现方案
参考 huggingface 的 transformers 中的实现思路来基于 paddle 进行开发

实现 tie_weight 函数步骤:
1. 获取模型 input embedding  权重对象 A
2. 获取模型 output embedding 权重对象 B
3. 让 A 和 B 都指向同一个权重值




## 命名与参数设计
参考：[飞桨 API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层 OP 设计
## API 实现方案

# 六、测试和验收的考量
参考：[新增 API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

测试 tie_weight 有两个办法:
* 直接判断输出层 weight 和输入层 weight 的 id，如果一致即通过，否则 Failed.
* 训练几个 step，经过几个反向后，看下输出层 weight 和输入层 weight 是否一致，如果一致即通过，否则 Failed.

用过 id 的一致性判断是否绑定成功, 简单高效,后面准备采用这种方式进行单侧:
构建单元测试, 测试模型的 get_input_embeding 得到的权重的 id 和 get_output_embeding 得到的权重 id 是都一致, 如果是一致就通过,都则不通过



# 七、可行性分析和排期规划

设计一个小脚本验证一下这种方式的有效性:
```python
import numpy as np
from paddle.nn import Embedding

"""step1 定义两个不同的embedding 对象 AA 和 BB"""
print('------------step1')
AA = Embedding(1,2)
BB = Embedding(1,2)

AA.weight = BB.weight # 进行权重的绑定

""" step2 测试一下绑定结果"""
print('------------step2')
print('检测 AA 和 BB 的id是否一致:', AA is BB,id(AA), id(BB))                               # AA 和 BB 的id 不一致
print('检测 AA.weight 和 BB.weight 的id是否一致:',AA.weight is BB.weight,id(AA.weight), id(BB.weight))   # 但是AA.weight 和 BB.weight 的id是一致的

print("AA.weight: ",AA.weight)
print("BB.weight: ",BB.weight)



""" step3 尝试修改一下AA的weight的值 BB的weight的值是否也跟着会一起修改"""
# 修改一下其中一个AA 的权重值, 看一下 BB的权重值会不会变化
print('------------step3')
AA.weight.set_value(np.array([[4.0,6.0]],dtype=np.float32))

print('检测 修改后的 AA.weight 和 BB.weight 的id是否一致:',AA.weight is BB.weight,id(AA.weight), id(BB.weight)) # AA.weight 和 BB.weight 的id是一致的
print("AA.weight 修改后的值: ",AA.weight)
print("BB.weight:",BB.weight)

```

时间和开发排期规划，主要 milestone
- 3.10 跟官方确认好开发思路
- 3.17 提交实现代码

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料
