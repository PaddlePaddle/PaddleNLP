========
FasterGeneration加速生成API
========

FasterGeneration是PaddleNLP v2.2版本加入的一个高性能推理功能，可实现基于CUDA的序列解码。该功能可以用于多种生成类的预训练NLP模型，例如GPT、BART、UnifiedTransformer等，并且支持多种解码策略。因此该功能主要适用于机器翻译，文本续写，文本摘要，对话生成等任务。

功能底层依托于 `FasterTransformer <https://github.com/NVIDIA/FasterTransformer>`_ ，该库专门针对Transformer系列模型及各种解码策略进行了优化。功能顶层封装于 `model.generate` 函数。功能的开启和关闭通过传入 `use_faster` 参数进行控制（默认为开启状态）。该功能具有如下特性：

- 全面支持生成式预训练模型。包括GPT、BART、mBART、UnifiedTransformer和UNIMO-text。
- 支持大多数主流解码策略。包括Beam Search、Sampling、Greedy Search。以及Diverse Sibling Search、Length Penalty等子策略。
- 解码速度快。最高可达非加速版generate函数的 **17倍**。HuggingFace generate函数的 **8倍**。**并支持FP16混合精度计算**。 详细性能试验数据请参见 `FasterGeneration Performence <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/faster_generation/perf>`_ 。
- 易用性强。功能的入口为 `model.generate` ，与非加速版生成api的使用方法相同，当满足加速条件时使用jit即时编译高性能算子并用于生成，不满足则自动切换回非加速版生成api。下图展示了FasterGeneration的启动流程：

.. image:: ../../imgs/faster_generation.png

快速开始
-----------

为体现FasterGeneration的易用性，我们在 `samples <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/experimental/faster_generation/samples>`_ 文件夹中内置了几个典型任务示例，下面以基于GPT模型的中文文本续写任务为例：

.. code-block::

    python samples/gpt_sample.py


如果是第一次执行，PaddleNLP会启动即时编译（ `JIT Compile <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/new_op/new_custom_op_cn.html#jit-compile>`_ ）自动编译高性能解码算子。

.. code-block::

    ...
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


关于该方法的更多参数可以参考API文档 `generate <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.generation_utils.html>`_ 。

`samples <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/experimental/faster_generation/samples>`_ 文件夹中的其他示例的使用方法相同。

其他示例
-----------

除了以上简单示例之外，PaddleNLP的examples中所有使用了 `model.generate()` 的示例都可以通过调整到合适的参数使用高性能推理。具体如下：

- `examples/dialogue/unified_transformer <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer>`_
- `model_zoo/gpt/faster_gpt <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt/faster_gpt>`_
- `examples/text_generation/unimo-text <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text>`_
- `examples/text_summarization/bart <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_summarization/bart>`_

根据提示修改对应参数即可使用FasterGeneration加速生成。下面我们以基于 `Unified Transformer` 的任务型对话为例展示一下FasterGeneration的加速效果：

打开以上链接中Unified Transformer对应的example，找到README中对应预测的脚本。稍作修改如下：

.. code-block::

    export CUDA_VISIBLE_DEVICES=0
        python infer.py \
        --model_name_or_path=unified_transformer-12L-cn-luge \
        --output_path=./predict.txt \
        --logging_steps=10 \
        --seed=2021 \
        --max_seq_len=512 \
        --max_knowledge_len=256 \
        --batch_size=4 \
        --min_dec_len=1 \
        --max_dec_len=64 \
        --num_return_sequences=1 \
        --decode_strategy=sampling \
        --top_k=5 \
        --device=gpu

由于这里只是展示性能，我们直接在 `model_name_or_path` 填入PaddleNLP预训练模型名称 `unified_transformer-12L-cn-luge` 。

可以看到，由于该任务为对话任务，我们为了防止模型生成过多安全回复（如：哈哈哈、不错等），保证生成结果具有更多的随机性，我们选择TopK-sampling作为解码策略，并让k=5。

打开 `infer.py` ，可以看到我们传入的脚本参数大多都提供给了 `model.generate()` 方法：

.. code-block::

    output = model.generate(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        seq_len=seq_len,
        max_length=args.max_dec_len,
        min_length=args.min_dec_len,
        decode_strategy=args.decode_strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        num_return_sequences=args.num_return_sequences,
        use_fp16_decoding=args.use_fp16_decoding,
        use_faster=args.faster)

运行脚本，输出结果如下：

.. code-block::

    step 10 - 1.695s/step
    step 20 - 1.432s/step
    step 30 - 1.435s/step

可以看到，非加速版 `generate()` 方法的预测速度为每个step耗时1.5秒左右。

下面我们在启动脚本中传入 `--faster` 参数，这会让 `generate()` 方法传入 `use_faster=True` ，启动加速模式。同时我们需要设置 `--min_dec_len=0` ，因为FasterGeneration当前还不支持该参数。新的脚本启动参数如下：

.. code-block::

    export CUDA_VISIBLE_DEVICES=0
        python infer.py \
        --model_name_or_path=unified_transformer-12L-cn-luge \
        --output_path=./predict.txt \
        --logging_steps=10 \
        --seed=2021 \
        --max_seq_len=512 \
        --max_knowledge_len=256 \
        --batch_size=4 \
        --min_dec_len=0 \
        --max_dec_len=64 \
        --num_return_sequences=1 \
        --decode_strategy=sampling \
        --top_k=5 \
        --device=gpu \
        --faster

再次运行脚本，输出结果如下（由于我们已经编译过高性能算子，所以这里不会重新编译）：

.. code-block::

    [2021-11-23 13:38:09,200] [   DEBUG] - skipping 'FasterTransformer' extension (up-to-date) build
    step 10 - 0.511s/step
    step 20 - 0.343s/step
    step 30 - 0.419s/step

可以看到，FasterGeneration的预测速度为每个step耗时0.4秒左右，提速超过三倍。如果减少 `num_return_sequences` ，可以得到更高的加速比。
