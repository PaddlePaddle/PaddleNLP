========
FasterGeneration加速生成API
========

FasterGeneration是PaddleNLP v2.2版本加入的一个高性能推理功能，可实现基于CUDA的序列解码。该功能可以用于多种生成类的预训练NLP模型，例如GPT、BART、UnifiedTransformer等，并且支持多种解码策略。因此该功能主要适用于机器翻译，文本续写，文本摘要，对话生成等任务。

功能底层依托于 `FasterTransformer <https://github.com/NVIDIA/FasterTransformer>`_ ，该库专门针对Transformer系列模型及各种解码策略进行了优化。功能顶层封装于 `model.generate` 函数。功能的开启和关闭通过传入 `use_faster` 参数进行控制（默认为开启状态）。该功能具有如下特性：

- 全面支持生成式预训练模型。包括GPT、BART、mBART、UnifiedTransformer和UNIMO-text。
- 支持大多数主流解码策略。包括Beam Search、Sampling、Greedy Search。以及Diverse Sibling Search、Length Penalty等子策略。
- 解码速度快。最高可达非加速版generate函数的 **17倍**。HuggingFace generate函数的 **8倍**。**并支持FP16混合精度计算**。 详细性能试验数据请参见 `FasterGeneration Performence <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/experimental/faster_generation/perf>`_ 。
- 易用性强。功能的入口为 `model.generate` ，与非加速版生成api的使用方法相同，当满足加速条件时使用jit即时编译高性能算子并用于生成，不满足则自动切换回非加速版生成api。

快速开始
-----------

为体现FasterGeneration的易用性，我们在 `samples` 文件夹中内置了几个典型任务示例，下面以基于GPT模型的中文文本续写任务为例：

.. code-block::

    python samples/gpt_sample.py


如果是第一次执行，PaddleNLP会启动即时编译（ `JIT Compile <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op_cn.html#jit-compile>`_ ）自动编译高性能解码算子。

.. code-block::

    ...
    Compiling user custom op, it will cost a few seconds.....
    2021-11-17 13:42:56,771 - INFO - execute command: cd /10.2/hub/PaddleNLP/paddlenlp/ops/extenstions && /usr/local/bin/python FasterTransformer_setup.py build
    INFO:utils.cpp_extension:execute command: cd /10.2/hub/PaddleNLP/paddlenlp/ops/extenstions && /usr/local/bin/python FasterTransformer_setup.py build
    grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
    running build
    running build_ext
    -- The C compiler identification is GNU 8.2.0
    -- The CXX compiler identification is GNU 8.2.0
    -- The CUDA compiler identification is NVIDIA 10.2.89
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /usr
    ...


编译过程通常会花费几分钟的时间但是只会进行一次，之后再次使用高性能解码不需要重新编译了。编译完成后会继续运行，可以看到生成的结果如下：

.. code-block::

    Model input: 花间一壶酒，独酌无相亲。举杯邀明月，
    Result: 对影成三人。

打开示例代码 `samples/gpt_sample.py` ，我们可以看到如下代码：

.. code-block::

    ...
    model = GPTLMHeadModel.from_pretrained(model_name)
    ...
    outputs, _ = model.generate(
        input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search')
    ...

可以看到，FasterGeneration的使用方法与 `model.generate()` 相同，只需传入输入tensor和解码相关参数即可，使用非常简便。如果要使用非加速版的 `model.generate()` 方法，只需传入 `use_faster=False` 即可，示例如下：

.. code-block::

    ...
    outputs, _ = model.generate(
        input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search', use_faster=False)
    ...

.. note::

    需要注意的是，如果传入 `model.generate()` 的参数不满足高性能版本的要求。程序会做出提示并自动切换为非加速版本，例如我们传入 `min_length=1` ，会得到如下提示：

    .. code-block::

        [2021-11-17 14:21:06,132] [ WARNING] - 'min_length != 0' is not supported yet in the faster version
        [2021-11-17 14:21:06,132] [ WARNING] - FasterGeneration is not available, and the original version would be used instead.


关于该方法的详细介绍可以参考 `generate <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.generation_utils.html>`_ 。

`samples` 文件夹中的其他示例的使用方法相同。

其他示例
-----------

除了以上简单示例之外，PaddleNLP的examples中所有使用了 `model.generate()` 的示例都可以通过调整到合适的参数使用高性能推理。具体如下：

- `examples/dialogue/unified_transformer <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer>`_
- `examples/language_model/gpt/faster_gpt <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/gpt/faster_gpt>`_
- `examples/text_generation/unimo-text <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text>`_
- `examples/text_summarization/bart <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_summarization/bart>`_

根据提示修改对应参数即可使用FasterGeneration加速生成。
