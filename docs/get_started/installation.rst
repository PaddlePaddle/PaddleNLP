安装PaddleNLP
============
以下安装过程默认用户已安装好paddlepaddle-gpu或paddlepaddle(版本大于或等于3.0)，paddlepaddle安装方式参照 飞桨官网_。

.. _飞桨官网: https://www.paddlepaddle.org.cn/

pip安装
--------
.. code-block:: bash

  pip install --upgrade paddlenlp

使用Anaconda或Miniconda3安装
--------
Anaconda 是一个开源的Python发行版本，旨在简化科学计算（数据科学、机器学习应用、大数据处理和可视化等）的工作流程。它包含了Python本身以及超过180个预编译的科学包和它们的依赖项。这些包涵盖了数据分析、机器学习、可视化等多个方面，使得用户无需再单独安装这些包及其复杂的依赖关系。
Miniconda 是Anaconda的一个轻量级版本，它只包含了conda、Python和一些少量的其他包。相比于Anaconda，Miniconda的体积更小，安装速度更快，更适合那些只需要特定包的用户，或者希望手动管理安装包的用户。


1、Windows安装
>>>>>>>>>

第一步 下载
:::::::::
* 在 Anaconda官网_ 或者 Miniconda官网_ 选择下载Windows 64-Bit版本。

.. _Anaconda官网: https://www.anaconda.com/download/success
.. _Miniconda官网: https://docs.anaconda.com/miniconda/

* 确保已经安装Visual C++ Build Tools(可以在开始菜单中找到)，如未安装，请点击 下载安装_。

.. _下载安装: https://go.microsoft.com/fwlink/?Linkid=691126

第二步 安装
:::::::::
运行下载的安装包(以.exe为后辍)，根据引导完成安装, 用户可自行修改安装目录。

第三步 使用
:::::::::
* 点击系统Windows图标，打开：所有程序->Anaconda3（64-bit）->Anaconda Prompt 或 Miniconda3（64-bit）->Anaconda Prompt
* 在命令行中执行下述命令

.. code-block:: bash

  # 创建名为my_paddlenlp的环境，指定Python版本为3.9或3.10
  conda create -n my_paddlenlp python=3.9
  # 进入my_paddlenlp环境
  conda activate my_paddlenlp
  # 安装PaddleNLP
  pip install --upgrade paddlenlp
  # 或者安装develop版本
  # pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

按如上方式配置后，即可在环境中使用PaddleNLP了，命令行输入python回车后，import paddlenlp试试吧，之后再次使用都可以通过打开'所有程序->Anaconda3/2（64-bit）->Anaconda Prompt'，再执行conda activate my_paddlenlp进入环境后，即可再次使用PaddleNLP。

2、Linux/Mac安装
>>>>>>>>>

第一步 下载
:::::::::
在 Anaconda官网_ 或者 Miniconda官网_ 选择下载对应系统版本下载（Mac下载Command Line Installer版本即可)。

.. _Anaconda官网: https://www.anaconda.com/download/success
.. _Miniconda官网: https://docs.anaconda.com/miniconda/

第二步 安装
:::::::::
打开终端，在终端安装Anaconda

.. code-block:: bash

  # Anaconda3-xxxx-Linux/MacOSX-x86_64/arm64.sh即下载的文件
  bash Anaconda3-xxxx-Linux/MacOSX-x86_64/arm64.sh
  
安装过程中一直回车即可，如提示设置安装路径，可根据需求修改，一般默认即可。

第三步 使用
:::::::::

.. code-block::bash

  # 创建名为my_paddlenlp的环境，指定Python版本为3.9或3.10
  conda create -n my_paddlenlp python=3.9
  # 进入my_paddlenlp环境
  conda activate my_paddlenlp
  # 安装PaddleNLP
  pip install --upgrade paddlenlp
  # 或者安装develop版本
  # pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

按如上方式配置后，即可在环境中使用PaddleNLP了，命令行输入python回车后，import paddlenlp试试吧，之后再次使用都可以通过打开终端，再执行conda activate my_paddlenlp进入环境后，即可再次使用PaddleNLP。

代码安装
---------
若需要从源码安装PaddleNLP，请克隆GitHub仓库并按照以下步骤操作：

.. code-block:: bash

  git clone https://github.com/PaddlePaddle/PaddleNLP.git
  cd PaddleNLP
  git checkout develop

使用Docker镜像体验PaddleNLP
^^^^^^^^

如果您没有Docker运行环境，请参考 `Docker官网`_ 进行安装

.. _Docker官网: https://www.docker.com

PaddleNLP基于PaddlePaddle提供的docker镜像进行docker使用，需参考 Paddle官网_ 拉去对应docker镜像后启动镜像实例，再通过命令安装PaddleNLP。

.. code-block:: bash
  # 安装PaddleNLP
  pip install --upgrade paddlenlp
  # 或者安装develop版本
  # pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html