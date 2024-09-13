安装PaddleNLP
============

以下安装过程默认用户已安装好paddlepaddle-gpu或paddlepaddle（版本大于或等于3.0）。PaddlePaddle的安装方式请参照`飞桨官网 <https://www.paddlepaddle.org.cn/>`_。

使用pip安装

-------

.. code-block:: bash

  pip install --upgrade paddlenlp

使用Anaconda或Miniconda3安装

----------------

Anaconda 是一个开源的Python发行版本，旨在简化科学计算（数据科学、机器学习应用、大数据处理和可视化等）的工作流程。它包含了Python本身以及超过180个预编译的科学包和它们的依赖项。这些包涵盖了数据分析、机器学习、可视化等多个方面，使得用户无需再单独安装这些包及其复杂的依赖关系。
Miniconda 是Anaconda的一个轻量级版本，它只包含了conda、Python和一些少量的其他包。相比于Anaconda，Miniconda的体积更小，安装速度更快，更适合那些只需要特定包的用户，或者希望手动管理安装包的用户。


1. **下载并安装**

   请前往`Anaconda官网 <https://www.anaconda.com/download/success>`_或`Miniconda官网 <https://docs.anaconda.com/miniconda/>`_选择对应系统的版本进行下载并安装。


2. **创建并激活环境**

   打开终端（Windows为Anaconda Prompt），执行以下命令：

   .. code-block:: bash

     conda create -n my_paddlenlp python=3.9  # 创建名为my_paddlenlp的环境，指定Python版本
     conda activate my_paddlenlp  # 激活环境
     pip install --upgrade paddlenlp  # 安装PaddleNLP

   若需要安装开发版本，请将上述安装命令替换为：

   .. code-block:: bash

     pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html


3. **验证安装**

   在激活的环境中，输入`python`回车后，尝试`import paddlenlp`以验证安装是否成功。

从源码安装

-------

若需要从源码安装PaddleNLP，请克隆GitHub仓库并按照以下步骤操作：

.. code-block:: bash

  git clone https://github.com/PaddlePaddle/PaddleNLP.git  # 克隆仓库
  cd PaddleNLP  # 进入目录
  git checkout develop  # 切换到develop分支（可选）
  pip install -e .  # 从源码安装

使用Docker镜像体验PaddleNLP

---------------

若您没有Docker运行环境，请先前往`Docker官网 <https://www.docker.com>`_进行安装。然后，基于`PaddlePaddle提供的docker镜像<https://www.paddlepaddle.org.cn/>`_拉取对应镜像并启动实例，再通过命令安装PaddleNLP。