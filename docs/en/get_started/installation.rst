Install PaddleNLP
~~~~~~~~~~~~~~~

The following guide will walk you through the installation process to ensure a smooth setup. Please note that this installation assumes you have already installed `paddlepaddle-gpu` or `paddlepaddle` (version >=3.0). If you haven't installed PaddlePaddle yet, please refer to the `PaddlePaddle official website`_.

.. _PaddlePaddle official website: https://www.paddlepaddle.org.cn/

For cuda12.3 and cuda11.8 environments, you may reference these installation commands:

.. code-block:: bash

  python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

pip Installation
--------

The simplest and quickest installation method is via pip. Just execute the following command in your terminal:

.. code-block:: bash

  pip install --upgrade --pre paddlenlp==3.0.0b4

This will automatically install the latest version of PaddleNLP.

Installation via Anaconda or Miniconda
--------------------------

Anaconda and Miniconda are popular Python distributions that simplify package management and environment configuration.

**Windows Installation Steps**:
^^^^^^^^^^^^^^^^^^^^^

1. **Download**: Visit the `Anaconda official website`_ or `Miniconda official website`_ to download the Windows 64-Bit installer.

.. _`Anaconda official website`: https://www.anaconda.com/download/success
.. _`Miniconda official website`: https://docs.conda.io/en/latest/miniconda.html
2. **Installation**: Run the downloaded installer and follow the on-screen instructions to complete the installation.

3. **Environment Configuration**:

   - Open "Anaconda Prompt" or "Miniconda Prompt".
   - Create a new environment and install PaddleNLP:

    .. code-block:: bash

      # Create environment named my_paddlenlp with Python 3.9 or 3.10
      conda create -n my_paddlenlp python=3.9
      # Activate environment
      conda activate my_paddlenlp
      # Install PaddleNLP
      pip install --upgrade --pre paddlenlp

    Or for nightly build installation:

    .. code-block:: bash

      pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

   Now you can use PaddleNLP in this environment.

**Linux/Mac Installation Steps**:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Download**: Visit `Anaconda Official Site`_ or `Miniconda Official Site`_ to download the installer for Linux/Mac OS.

.. _`Anaconda Official Site`: https://www.anaconda.com/download/success
.. _`Miniconda Official Site`: https://docs.anaconda.com/miniconda/
2. **Installation**: Open the terminal, navigate to the directory where the file was downloaded, and execute the installation script.

3. **Environment Configuration**:

   - Create a new environment and install PaddleNLP, following the same steps as for Windows.

Code Installation
-----------------

If you prefer to install PaddleNLP from source code, you can do so by cloning the GitHub repository:

.. code-block:: bash

  git clone https://github.com/PaddlePaddle/PaddleNLP.git
  cd PaddleNLP
  git checkout develop

Then follow the instructions in the repository for subsequent installation steps.

Using Docker Images to Experience PaddleNLP
------------------------------------------

If you want to experience PaddleNLP in an isolated environment, you can use Docker. First ensure Docker is installed. Then you can pull the Docker image provided by PaddlePaddle and install PaddleNLP within it:

.. code-block:: bash

  # Assuming you've already pulled the PaddlePaddle Docker image
  # After entering the Docker container
  pip install --upgrade --pre paddlenlp

Alternatively, to install the development version:

.. code-block:: bash

  pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

This allows you to easily use PaddleNLP within the Docker container.