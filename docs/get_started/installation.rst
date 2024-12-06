安装PaddleNLP
~~~~~~~~~~~~~~~

以下指南将引导您完成安装过程，确保您能够轻松上手。请注意，本安装过程假设您已安装好`paddlepaddle-gpu`或`paddlepaddle`（版本大于或等于3.0）。如果您尚未安装PaddlePaddle，请参考 `飞桨官网`_ 进行安装。

.. _飞桨官网: https://www.paddlepaddle.org.cn/

pip安装
--------

最简单快捷的安装方式是使用pip。只需在命令行（终端）中运行以下命令：

.. code-block:: bash

  pip install --upgrade --pre paddlenlp

这将会自动安装最新版本的PaddleNLP。

使用Anaconda或Miniconda安装
--------------------------

Anaconda和Miniconda是流行的Python发行版本，它们能够简化包管理和环境配置。


**Windows安装步骤**：
^^^^^^^^^^^^^^^^^^^^^

1. **下载**：访问 `Anaconda官网`_ 或 `Miniconda官网`_，下载适用于Windows 64-Bit的安装包。

.. _`Anaconda官网`: https://www.anaconda.com/download/success
.. _`Miniconda官网`: https://docs.anaconda.com/miniconda/

2. **安装**：运行下载的安装包，按照屏幕上的指示完成安装。

3. **配置环境**：

   - 打开“Anaconda Prompt”或“Miniconda Prompt”。
   - 创建一个新的环境并安装PaddleNLP：

    .. code-block:: bash

      # 创建名为my_paddlenlp的环境，指定Python版本为3.9或3.10
      conda create -n my_paddlenlp python=3.9
      # 激活环境
      conda activate my_paddlenlp
      # 安装PaddleNLP
      pip install --upgrade --pre paddlenlp

    或者，如果您想安装nightly版本：

    .. code-block:: bash

      pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

   现在，您可以在这个环境中使用PaddleNLP了。


**Linux/Mac安装步骤**：
^^^^^^^^^^^^^^^^^^^^^

1. **下载**：访问 `Anaconda官网`_ 或 `Miniconda官网`_，下载适用于Linux/Mac操作系统的安装包。

.. _`Anaconda官网`: https://www.anaconda.com/download/success
.. _`Miniconda官网`: https://docs.anaconda.com/miniconda/

2. **安装**：打开终端，导航到下载文件的目录，并执行安装脚本。

3. **配置环境**：

   - 创建一个新的环境并安装PaddleNLP，步骤与Windows相同。

代码安装
--------

如果您希望从源代码安装PaddleNLP，可以通过克隆GitHub仓库来实现：

.. code-block:: bash

  git clone https://github.com/PaddlePaddle/PaddleNLP.git
  cd PaddleNLP
  git checkout develop

然后，您可以按照仓库中的说明进行后续安装步骤。

使用Docker镜像体验PaddleNLP
-------------------

如果您想在一个隔离的环境中体验PaddleNLP，可以使用Docker。首先，请确保您已安装Docker。然后，您可以拉取PaddlePaddle提供的Docker镜像，并在其中安装PaddleNLP：

.. code-block:: bash

  # 假设您已经拉取了PaddlePaddle的Docker镜像
  # 进入Docker容器后
  pip install --upgrade --pre paddlenlp

或者，如果您想安装开发版本：

.. code-block:: bash

  pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

这样，您就可以在Docker容器中轻松使用PaddleNLP了。