==========================================
贡献优质的模型权重
==========================================

1. 模型网络结构种类
------------------------------------------
PaddleNLP已支持绝大部分主流的预训练模型网络，既包括百度自研的预训练模型（如ERNIE系列），
也涵盖业界主流的预训练模型（如BERT，GPT，RoBERTa，XLNet等）。

下面列出了PaddleNLP目前支持的15类网络结构（持续增加中，也非常欢迎你的贡献）：

- ALBERT
- BERT
- BigBird
- DistilBert
- ELECTRA
- ERNIE
- ERNIE-GEN
- ERNIE-GRAM
- GPT
- NeZha
- RoBERTa
- SKEP
- TinyBert
- UnifiedTransformer
- XLNet

2. 模型参数权重类型
------------------------------------------
我们非常欢迎大家贡献优质模型参数权重。
参数权重类型包括但不限于（以BERT模型网络为例）：

1. PaddleNLP还未收录的BERT预训练模型参数权重（如）
2. BERT模型在其他垂类领域（如法律，医学等）的预训练模型参数权重（如）
3. 基于BERT在下游具体任务进行fine-tuning后的模型参数权重（如）
4. 其他任何你觉得有价值的模型参数权重

3. 参数权重格式转换
------------------------------------------
当我们想要贡献github上开源的某模型权重时，但是发现该权重保存为其他的深度学习框架（PyTorch，TensorFlow等）的格式，
这刘需要我们进行不同深度学习框架间的模型格式转换，下面的链接给出了一份详细的关于Pytorch到Paddle模型格式转换的教程。
`Pytorch到Paddle模型格式转换文档 <./convert_pytorch_to_paddle.rst>`_

4. 准备模型权重所需文件
------------------------------------------
一般来说，我们需要提供 ``model_config`` ，``model_state`` ，``tokenizer_config`` 以及 ``vocab`` 这四个文件
才能完成一个模型的贡献。

对于 ``bert-base-uncased-sst-2-finetuned`` 这个模型来说，
我们需要提供的文件如下：

1. model_config.json
2. model_state.pdparams
3. tokenizer_config.json
4. vocab.txt

5. 进行贡献
------------------------------------------
在准备好模型权重贡献所需要的文件后，我们就可以开始我们的贡献了。
我们是通过在github上提PR（pull request）进行贡献。

1. 如果你是首次贡献模型，你需要在 ``PaddleNLP/paddlenlp/transformers/community/`` 下新建一个目录，
目录名称使用你的github名称，如github名为 ``yingyibiao`` ，
则新建目录 ``PaddleNLP/paddlenlp/transformers/community/yingyibiao/`` 。

2. 接着在你的目录下新建一个模型目录，目录名称为本次贡献的模型名称，如我想贡献 ``bert-base-uncased-sst-2-finetuned`` 这个模型，
则新建目录 ``PaddleNLP/paddlenlp/transformers/community/yingyibiao/bert-base-uncased-sst-2-finetuned/``。

3. 在步骤2的目录下加入两个文件，分别为 ``README.md`` 和 ``files.json`` 。
``README.md`` 为对你贡献的模型的详细介绍，``files.json`` 为模型权重所需文件以所对应存储路径。

第一次进行开源贡献的同学可以参考 `first-contributions <https://github.com/firstcontributions/first-contributions>`_。

模型权重贡献示例请参考 `xxxModel PR示例 <.>`_