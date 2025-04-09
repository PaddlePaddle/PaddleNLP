========================
分享你的数据集
========================

除了使用PaddleNLP内置的数据集以外，我们也鼓励用户向PaddleNLP贡献自己的数据集。

下面我们来介绍一下贡献数据集的详细流程：

配置环境
---------------

#. 编写和测试PaddleNLP代码需要依赖python3.6以上版本以及最新版本的PaddlePaddle。请确保正确安装以上依赖。
#. 在PaddleNLP的github页面上点击Fork按钮，在自己的github中创建一份PaddleNLP repo的副本。
#. 将您frok的内容下载到本地，并将官方repo作为remote。

   .. code-block::

       git clone https://github.com/USERNAME/PaddleNLP
       cd PaddleNLP
       git remote add upstream https://github.com/PaddlePaddle/PaddleNLP.git

#. 安装pre-commit钩子，它可以帮助我们格式化源代码，再提交前自动检查代码问题。不满足钩子的PR **不能** 被提交到PaddleNLP。

   .. code-block::

       pip install pre-commit
       pre-commit install

添加一个 :class:`DatasetBuilder` 
----------------------------------

#. 创建一个新的本地分支，一般从develop 分支上创建新分支。

   .. code-block::

       git checkout -b my-new-dataset

#. 找到您本地repo下的 `PaddleNLP/paddlenlp/datasets/` 路径，PaddleNLP的所有数据集代码都储存在这个文件夹下。

   .. code-block::

       cd paddlenlp/datasets

#. 为您的数据集确定一个 `name`，例如 `squad` , `chnsenticorp` 等，这个 `name` 就是您的数据集被读取时的名称。
    
   .. note::

       - 为了方便别人使用您的数据集，确保这个 `name` **不会太长而且能够正确的表义**。
       - 数据集的 `name` 格式应为snake case。

#. 在该路径下创建python文件，文件名是数据集的 `name`，例如 `squad.py` 。并在这个文件中编写数据集的 :class:`DatasetBuilder` 代码。

   :class:`DatasetBuilder` 的编写可以参考教程 :doc:`如何创建一个DatasetBuilder <./how_to_write_a_DatasetBuilder>` 。里面给出了详细的步骤和规范。

   我们也推荐您参考已有数据集的 :class:`DatasetBuilder` 进行创建，从已有代码copy一些共用部分可能对您编写自己的数据集代码有所帮助，下面是一些已有数据集的示例：

   -  `iwslt15.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/iwslt15.py>`__ 翻译数据集，包含词表文件。
   -  `glue.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/glue.py>`__ glue数据集，包含多个子数据集，文件格式为tsv。
   -  `squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/squad.py>`__ 阅读理解数据集，文件格式为json。
   -  `imdb.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/imdb.py>`__ imdb数据集，每个split包含多个文件。
   -  `ptb.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/ptb.py>`__ 语料库数据集。
   -  `msra_ner.py <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/msra_ner.py>`__ 序列标注数据集。

#. 开发完成后，可以使用 :attr:`load_dataset` 测试您创建的数据集中的split能否正确被识别。也可以使用 :attr:`print` 看看数据集读入的格式是否符合您的预期：

   .. code-block::

       from paddlenlp.datasets import load_dataset

       ds = load_dataset('your_dataset_name', splits='your_split')
       print(ds[0])

提交您的成果
---------------

#. 当您认为数据集的代码已经ready后，就可以在本地commit您的修改了：
   
   .. code-block::
       
       git add PaddleNLP/paddlenlp/datasets/your_dataset_name.py
       git commit

#. 在提交修改之前，最好获取获取先upstream的最新代码并更新当前分支。

   .. code-block::
       
       git fetch upstream
       git pull upstream develop

#. 将本地的修改推送到GitHub上，并在GitHub上向PaddleNLP提交Pull Request。

   .. code-block::
       
       git push origin my-new-dataset

以上就是像PaddleNLP贡献数据集的完整流程了。我们看到您的PR后会尽快review，如果有任何问题都会尽快反馈给您。如果没有问题的话我们就会合入到PaddleNLP repo，您贡献的数据集就可以供其他人使用啦。

如果您对贡献数据集还有任何疑问，欢迎加入官方QQ技术交流群: 973379845向我们提出。我们会尽快为您解答。