# Few-Shot Learning(FSL)
Few-Shot Learning 旨在研究如何从少量有监督的训练样本中学习出具有良好泛化性的模型，对训练数据很少或监督数据获取成本极高的应用场景有很大价值。

随着大规模预训练模型的不断涌现，FSL 结合预训练模型的先验知识和强大的泛化能力在下游任务效果上取得了显著提升，为大规模预训练模型结合 FSL 的工业落地应用带来了无限可能性。

我们旨在为 FSL 领域的研究者提供简单易用、全面、前沿的 FSL 策略库，便于研究者基于 FSL 策略库将注意力集中在算法创新上。我们会持续开源 FSL 领域的前沿学术工作，并在中文小样本学习测评基准 [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE) 上进行评测。

## Benchmark
我们在 FewCLUE 9 个任务的 test_public.json 测试集上进行了效果评测

| 算法 | 预训练模型  | Score  | eprstmt  | bustm  | ocnli  | csldcp  | tnews  |  cluewsc | iflytek | csl | chid |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ | ------------ | ---------- |
| P-tuning  | ERNIE1.0  | 55.70 | 83.28  | 63.43  | 35.36  | 60.54  | 50.02  | 54.51  | 50.14 | 54.93 | 41.16 |
| EFL       | ERNIE1.0  | 54.47 | 84.10  | 60.10  | 35.12  | 56.61  | 56.57  | 53.59  | 46.37 | 61.21 | 36.56 |

## 策略库
- [P-tuning](./p-tuning)
- [EFL](./efl)
- PET(Todo)

## References
[1]X. Liu et al., “GPT Understands, Too,” arXiv:2103.10385 [cs], Mar. 2021, Accessed: Mar. 22, 2021. [Online]. Available: http://arxiv.org/abs/2103.10385
[2] Wang, Sinong, Han Fang, Madian Khabsa, Hanzi Mao, and Hao Ma. “Entailment as Few-Shot Learner.” ArXiv:2104.14690 [Cs], April 29, 2021. http://arxiv.org/abs/2104.14690.
