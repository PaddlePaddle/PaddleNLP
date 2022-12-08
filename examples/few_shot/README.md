# Few-Shot Learning (FSL)

Few-Shot Learning 旨在研究如何从少量有监督的训练样本中学习出具有良好泛化性的模型，对训练数据很少或监督数据获取成本极高的应用场景有很大价值。

随着大规模预训练模型的不断涌现，FSL 结合预训练模型的先验知识和强大的泛化能力在下游任务效果上取得了显著提升，为大规模预训练模型结合 FSL 的工业落地应用带来了无限可能性。

我们旨在为 FSL 领域的研究者提供简单易用、全面、前沿的 FSL 策略库，便于研究者基于 FSL 策略库将注意力集中在算法创新上。我们会持续开源 FSL 领域的前沿学术工作，并在中文小样本学习测评基准 [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE) 上进行评测。

## Benchmark
我们在 FewCLUE 9 个任务的 test_public.json 测试集上进行了效果评测

| 算法 | 预训练模型  | eprstmt  | csldcp  | iflytek  | tnews  | ocnli  |  bustm | chid | csl | cluewsc |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ | ------------ | ---------- |
| PET       | ERNIE-1.0-Large-CW  | 88.03  | 63.79  | 56.43  | 56.57  | 56.27  | 72.69  | 91.39 | 76.00 | 78.79 |
| P-Tuning  | ERNIE-1.0-Large-CW  | 89.84  | 64.57  | 45.80  | 57.41  | 44.13  | 68.51  | 90.00 | 74.67 | 73.26 |
| EFL       | ERNIE-1.0-Large-CW  | 90.82  | 54.48  | 46.71 | 54.43  | 43.17 | 72.63 | 85.71 | 61.52 | 80.02 |

**注释**:
- 表格中 CHID 数据集的指标与 FewCLUE 榜单指标计算方式不同。
- 由于 iflytek 和 csldcp 标签数较多，每条样本采样 5 个非正确标签作为负样本训练评测。
- 为统一配置，除 EFL-iflytek 外均训练 1000 steps，EFL-iflytek 训练 5000 steps。

## Models
- [P-tuning](./p-tuning)
- [EFL](./efl)
- [PET](./pet)

## References

- [1] X. Liu et al., “GPT Understands, Too,” arXiv:2103.10385 [cs], Mar. 2021, Accessed: Mar. 22, 2021. [Online]. Available: http://arxiv.org/abs/2103.10385.

- [2] Wang, Sinong, Han Fang, Madian Khabsa, Hanzi Mao, and Hao Ma. “Entailment as Few-Shot Learner.” ArXiv:2104.14690 [Cs], April 29, 2021. http://arxiv.org/abs/2104.14690.

- [3] Schick, Timo, and Hinrich Schütze. “Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference.” ArXiv:2001.07676 [Cs], January 25, 2021. http://arxiv.org/abs/2001.07676.
