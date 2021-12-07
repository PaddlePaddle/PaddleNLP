============
Transformer高性能加速
============


使用环境说明
------------

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* `FasterTransformer <https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup>`_ 使用必要的环境
* 环境依赖

  - attrdict
  - pyyaml

  .. code-block::

      pip install attrdict pyyaml


快速开始
------------

我们实现了基于 FasterTransformer 的自定义 op 的接入，用于加速文本生成模型在 GPU 上的预测性能。接下来，我们将分别介绍基于 Python 动态图和预测库使用 FasterTransformer 自定义 op 的方式，包括 op 的编译与使用。

Python 动态图使用自定义 op
------------

JIT 自动编译
^^^^^^^^^^^^

目前当基于动态图使用 FasterTransformer 预测加速自定义 op 时，PaddleNLP 提供了 Just In Time 的自动编译，在一些 API 上，用户无需关注编译流程，可以直接执行对应的 API，程序会自动编译需要的第三方库。

以 Transformer 为例，可以直接调用 `TransformerGenerator()` 这个 API，程序会自动编译。使用示例可以参考 `Transformer 预测加速使用示例-sample <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/faster_transformer/sample/decoding_sample.py>`_，`Transformer 预测加速使用示例-机器翻译 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer/faster_transformer>`_。

编译自定义OP
^^^^^^^^^^^^

除了自动编译外，如果需要自行编译，我们已经提供对应的 CMakeLists.txt，可以参考使用如下的方式完成编译。

PaddleNLP 准备
""""""""""""

首先，如果需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译:

以下以从 github 上 clone 一个新版 PaddleNLP 为例:

.. code-block::

    git clone https://github.com/PaddlePaddle/PaddleNLP.git

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

.. code-block::

    export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
    cd PaddleNLP/paddlenlp/ops/

编译
""""""""""""

编译之前，请确保安装的 PaddlePaddle 的版本高于 2.1.0 或是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

.. code-block::

    mkdir build
    cd build/
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.x
    make -j
    cd ../

可以使用的编译选项包括：

* `-DPY_CMD`: 指定当前装有 PaddlePaddle 版本的 python 环境，比如 `-DPY_CMD=python3.7`。若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python。
* `-DSM`: 是指的所用 GPU 的 compute capability，建议不使用该选项设置，未设置时将自动检测。如要设置，需根据 [compute capability](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 进行设置，如 V100 时设置 `-DSM=70` 或 T4 时设置 `-DSM=75`。
* `-DWITH_GPT`: 是否编译带有 GPT 相关的 lib。若使用 GPT-2 高性能推理，需要加上 `-DWITH_GPT=ON`。默认为 OFF。
* `-DWITH_UNIFIED`: 是否编译带有 Unified Transformer 或是 UNIMOText 相关的 lib。若使用，需要加上 `-DWITH_UNIFIED=ON`。默认为 ON。
* `-DWITH_BART`: 是否编译带有 BART 支持的相关 lib。若使用，需要加上 `-DWITH_BART=ON`。默认为 ON。
* `-DWITH_DECODER`: 是否编译带有 decoder 优化的 lib。默认为 ON。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 FasterTransformer decoding 执行的库。

使用 Transformer decoding 高性能推理
^^^^^^^^^^^^

编写 python 脚本的时候，调用 `FasterTransformer API <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.ops.faster_transformer.transformer.faster_transformer.html#paddlenlp.ops.faster_transformer.transformer.faster_transformer.FasterTransformer>`_ 即可实现 Transformer 模型的高性能预测。

举例如下：

.. code-block::

    from paddlenlp.ops import FasterTransformer

    transformer = FasterTransformer(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        topk=args.topk,
        topp=args.topp,
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding)

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以如前文所述进行编译。编译好后，使用 `FasterTransformer(decoding_lib="/path/to/lib", ...)` 可以完成导入。

更详细的例子可以参考 `Transformer 预测加速使用示例-sample <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/faster_transformer/sample/decoding_sample.py>`_，`Transformer 预测加速使用示例-机器翻译 <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer/faster_transformer>`_，我们提供了更详细用例。

Transformer decoding 示例代码
""""""""""""

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

.. code-block::

    export CUDA_VISIBLE_DEVICES=0
    export FLAGS_fraction_of_gpu_memory_to_use=0.1
    # 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
    ./build/third-party/build/fastertransformer/bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so

使用 PaddlePaddle 仅执行 decoding 测试（float16）：
执行 float16 的 decoding，需要在执行的时候，加上 `--use_fp16_decoding` 选项。

.. code-block::

    export CUDA_VISIBLE_DEVICES=0
    export FLAGS_fraction_of_gpu_memory_to_use=0.1
    # 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
    ./build/third-party/build/fastertransformer/bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so --use_fp16_decoding

其中，`decoding_gemm` 不同参数的意义可以参考 `FasterTransformer 文档 <https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos>`_。这里提前执行 `decoding_gemm`，可以在当前路径下生成一个 config 文件，里面会包含针对当前 decoding 部分提供的配置下，性能最佳的矩阵乘的算法，并在执行的时候读入这个数据。

C++ 预测库使用自定义 op
------------

编译自定义OP
^^^^^^^^^^^^

在 C++ 预测库使用自定义 OP 需要将实现的 C++、CUDA 代码**以及 C++ 预测的 demo**编译成一个可执行文件。因预测库支持方式与 Python 不同，这个过程将不会产生自定义 op 的动态库，将直接得到可执行文件。我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。并获取执行 demo。

PaddleNLP 准备
""""""""""""

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 FasterTransformer 相关 lib，需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译:

以下以从 github 上 clone 一个新版 PaddleNLP 为例:

.. code-block::

    git clone https://github.com/PaddlePaddle/PaddleNLP.git

其次，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

.. code-block::

    cd PaddleNLP/paddlenlp/ops/

编译
""""""""""""

编译之前，请确保安装的 PaddlePaddle 的版本高于 2.1.0 或是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

.. code-block::

    mkdir build
    cd build/
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=/path/to/paddle_inference_lib/ -DDEMO=./demo/transformer_e2e.cc -DON_INFER=ON -DWITH_MKL=ON
    make -j
    cd ../

可以使用的编译选项包括：

* `-DPADDLE_LIB`: 需要指明使用的 PaddlePaddle 预测库的路径 `/path/to/paddle_inference_install_dir/`，需要使用的 PaddlePaddle 的 lib 可以选择自行编译或者直接从官网下载 `paddle_inference_linux_lib <https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux>`_。需要注意的是，在该路径下，预测库的组织结构满足：
  .. code-block::

      .
      ├── CMakeCache.txt
      ├── paddle/
        ├── include/
        └── lib/
      ├── third_party/
        ├── cudaerror/
        ├── install/
        └── threadpool/
      └── version.txt

* `-DDEMO`: 说明预测库使用 demo 的位置。比如指定 -DDEMO=./demo/transformer_e2e.cc 或是 -DDEMO=./demo/gpt.cc。最好使用绝对路径，若使用相对路径，需要是相对于 `PaddleNLP/paddlenlp/ops/faster_transformer/src/` 的相对路径。
* `-DSM`: 是指的所用 GPU 的 compute capability，建议不使用该选项设置，未设置时将自动检测。如要设置，需根据 [compute capability](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 进行设置，如 V100 时设置 `-DSM=70` 或 T4 时设置 `-DSM=75`。
* `-DWITH_GPT`: 是否编译带有 GPT 相关的 lib。若使用 GPT-2 高性能推理，需要加上 `-DWITH_GPT=ON`。默认为 OFF。
* `-DWITH_UNIFIED`: 是否编译带有 Unified Transformer 或是 UNIMOText 相关的 lib。若使用，需要加上 `-DWITH_UNIFIED=ON`。默认为 ON。
* `-DWITH_BART`: 是否编译带有 BART 支持的相关 lib。若使用，需要加上 `-DWITH_BART=ON`。默认为 ON。
* `-DWITH_DECODER`: 是否编译带有 decoder 优化的 lib。默认为 ON。
* `-DWITH_MKL`: 若当前是使用的 mkl 的 Paddle lib，那么需要打开 MKL 以引入 MKL 相关的依赖。
* `-DON_INFER`: 是否编译 paddle inference 预测库。
* **当使用预测库的自定义 op 的时候，请务必开启 `-DON_INFER=ON` 选项，否则，不会得到预测库的可执行文件。**

执行 Transformer decoding on PaddlePaddle
""""""""""""

编译完成后，在 `build/bin/` 路径下将会看到 `transformer_e2e` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

.. code-block::

    cd bin/
    ./transformer_e2e -batch_size <batch_size> -gpu_id <gpu_id> -model_dir <model_directory> -vocab_file <dict_file> -data_file <input_data>

举例说明：

.. code-block::

    cd bin/
    # 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
    ../third-party/build/fastertransformer/bin/decoding_gemm 8 5 8 64 38512 256 512 0
    ./transformer_e2e -batch_size 8 -gpu_id 0 -model_dir ./infer_model/ -vocab_file DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 -data_file DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en

其中：

* `decoding_gemm` 不同参数的意义可以参考 `FasterTransformer 文档 <https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos>`_。这里提前执行 `decoding_gemm`，可以在当前路径下生成一个 config 文件，里面会包含针对当前 decoding 部分提供的配置下，性能最佳的矩阵乘的算法，并在执行的时候读入这个数据。
* `DATA_HOME` 则是 `paddlenlp.utils.env.DATA_HOME` 返回的路径。

预测所需要的模型文件，可以通过 `faster_transformer/README.md <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/transformer/faster_transformer/README.md>`_ 文档中所记述的方式导出。

