============
Introduction to Model Compression
============

In recent years, Transformer-based language models have achieved substantial progress in NLP tasks such as machine translation, reading comprehension, text matching, and natural language inference. However, the massive number of parameters and computational resource requirements make BERT and its variants challenging to deploy. The development of model compression techniques has alleviated these issues.

Overview of Model Compression
----------------------------

Model compression reduces model storage and accelerates inference speed while maintaining acceptable accuracy. Common model compression methods include model pruning, quantization, and distillation. Below is a brief introduction to these techniques.

Model Pruning
^^^^^^^^^^^^^
Model pruning removes unimportant network connections from a trained model to reduce redundancy and computational load, thereby decreasing model storage and significantly accelerating inference speed.

Quantization
^^^^^^^^^^^^^
Typically, neural network parameters are represented using 32-bit floating-point numbers. However, such high precision is often unnecessary. Quantization reduces model storage by using lower-precision representations (e.g., INT8 instead of Float32). For instance, Stochastic Gradient Descent (SGD) only requires 6-8 bits of precision. Thus, proper quantization can reduce model size while maintaining accuracy and enable efficient CPU execution. Common quantization methods include binary neural networks, ternary weight networks, and XNOR networks.

Distillation
^^^^^^^^^^^^^
Knowledge distillation involves training a student model (with fewer parameters) to mimic a teacher model (with more parameters). The student model learns from the teacher's knowledge, achieving better performance than standalone training. A typical approach is distilling BERT-base to Bi-LSTM or smaller BERT variants. For example, DistilBERT retains 97% of BERT-base's accuracy with 40% fewer parameters and 60% faster inference.

Model Compression Examples
--------------------------

Below are examples of model compression implemented with PaddlePaddle. The tutorial *Knowledge Distillation from BERT to Bi-LSTM* serves as a "Hello World" example for distillation. The *BERT Compression Using DynaBERT's Strategy* demonstrates DynaBERT, which trains multiple subnets of different sizes simultaneously, allowing direct model pruning during inference.

.. toctree::
   :maxdepth: 1

   distill_lstm.rst
   ofa_bert.rst