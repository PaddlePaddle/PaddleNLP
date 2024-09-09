# ERNIE-Health 中文医疗预训练模型

医疗领域存在大量的专业知识和医学术语，人类经过长时间的学习才能成为一名优秀的医生。那机器如何才能“读懂”医疗文献呢？尤其是面对电子病历、生物医疗文献中存在的大量非结构化、非标准化文本，计算机是无法直接使用、处理的。这就需要自然语言处理（Natural Language Processing，NLP）技术大展身手了。

## 模型介绍

本项目针对中文医疗语言理解任务，开源了中文医疗预训练模型 [ERNIE-Health](https://arxiv.org/pdf/2110.07244.pdf)（模型名称`ernie-health-chinese`）。

ERNIE-Health 依托百度文心 ERNIE 先进的知识增强预训练语言模型打造, 通过医疗知识增强技术进一步学习海量的医疗数据, 精准地掌握了专业的医学知识。ERNIE-Health 利用医疗实体掩码策略对专业术语等实体级知识学习, 学会了海量的医疗实体知识。同时，通过医疗问答匹配任务学习病患病状描述与医生专业治疗方案的对应关系，获得了医疗实体知识之间的内在联系。ERNIE-Health 共学习了 60 多万的医疗专业术语和 4000 多万的医疗专业问答数据，大幅提升了对医疗专业知识的理解和建模能力。此外，ERNIE-Health 还探索了多级语义判别预训练任务，提升了模型对医疗知识的学习效率。该模型的整体结构与 ELECTRA 相似，包括生成器和判别器两部分。

![Overview_of_EHealth](https://user-images.githubusercontent.com/25607475/163949632-8b34e23c-d0cd-49df-8d88-8549a253d221.png)

更多技术细节可参考论文
- [Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](https://arxiv.org/pdf/2110.07244.pdf)

详细请参考: https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/ernie-health
