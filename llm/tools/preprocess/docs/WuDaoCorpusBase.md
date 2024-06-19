# WuDaoCorpus2.0 Base 语料


| 名称 | 文本类型 | 纯文本大小 |
|-|-|-|
| WuDaoCorpus2.0 Base| 中文 | 200GB |

WuDaoCorpora是悟道爬取的中文大规模语料。整体数量为3TB，目前开源的部分为WuDaoCorpus2.0 bases数据集，大小为200GB。

## 数据获取

**1. 下载解压**

用户微信登录[官网](https://resource.wudaoai.cn/home)，即可直接下载数据。下载好的压缩数据约 64GB。解压
```
unrar x WuDaoCorpus2.0_base_200G.rar
```
**2. 语料分词**

由于WuDao数据集比较大，分词比较耗时，这里先进行了语料分词：
```shell
python words_segmentation.py \
    --input_path ./WuDaoCorpus2.0_base_200G \
    --workers 40  \
    --data_format wudao \
    --cn_seg_func seg \
    --output_path ./wudao_lac_cut \
```

注：预训练需要实现 SOP( Sentence Order Predict) 任务，在分词的同时，我们使用 简单规则 进行了文本断句。如果语料只有一句话，建议去除SOP loss，训练时设置 `binary_head=False`。

**3. 转换为jsonl格式**

文本转化完成后。我们使用 `../data_tools/trans_to_json.py`重新转换为jsonl格式（分词完毕）。
```shell
python ./trans_to_json.py  \
    --input_path ./wudao_lac_cut \
    --output_path wudao_corpus_200g.jsonl \
    --workers 40
```
在当前目录下产出数据`wudao_corpus_200g.jsonl`。格式如下：
```
{"text": "主持人 : 作为 一个 曲线救国 的 路线 我们 没 办法 。\n金鑫 : 考试 和 分数 只是 一个 阶段性 的 评价 手段 , 不是 目的 , 就 像 人 活着 的 目的 不是 为了 吃饭 , 吃饭 是 为了 让 我们 活下去 , 我们 学习 的 目的 不是 为了 考试 , 不是 为了 那个 分数 , 而是 我 掌握 了 知识 , 成为 我 内在 的 能力 , 将来 我 去 创作 创造 工作 , 我能 把 它 做 得 更好 。\n主持人 : 特别感谢 金总 今天 接受 我 的 访谈 , 也 让 我 从 别的 层面 看到 了 一对一 到底 存在 的 道理 是 什么 , 并且 能 发展 那么 好 的 原因 在 哪里 。\n在 节目 后 您 谈谈 您 对 一对一 未来 的 希望 , 包括 您 对 它 未来 的 设想 是 什么 ？\n金鑫 : 一对一 个性化 教育 现在 还是 在 初级阶段 , 如果 是 四个 阶段 的话 , 现在 还是 在 第一阶段 到 第二阶段 迈进 的 , 学大 在 这方面 我们 希望 能 做 得 更 快 更 远 一些 。\n将来 个性化 教育 一定 是 能够 帮助 学生 在 成绩 上 的 提升 , 能够 更好 的 成长 , 进而 成为 对 社会 对 国家 更 有用 的 人才 , 就是 我们 的 成绩 、 成长 、 成才 。\n学大 1 对 1 教育 的 教师 团队 由 各科 优秀教师 、 考试 指导 专家 、 心理 辅导 专家 及 学习 方法 指导 专家 组成 , 同时 配备 专职 班主任 及 学习 监管 师 , 全方位 辅导   顺利 而 有序 的 运作 。\n其中 部分 教师 担任 多年 毕业班 教学 工作 , 多次 参与 中 考试 命题 研究 及 阅卷 工作 , 深谙 中 考试 精髓 , 能够 在 短 的 时间 内 引领 学生 掌握 中 考试 知识   重点 , 快速 提分 。\n■   对于 成绩 差 的 学生 : 注重 学生 基础知识 , 力求 让 学生 在 基础 中 找 自信 , 在 自信 中 提升 ；\n注重 主观题 的 解题 方法 及 思路 , 以此 来 加强 对 基础知识 的 运用 。\n■   对于 成绩 需要 拔高 的 学生 : 找出 学生 弱点 , 加强 基础 , 重点 提高 弱势 项目 。\n"}
{"text": "武田信玄 是 天生 的 武将 , 一生 开拓 了 八十五万 石至 九十余万 石之多 的 领地 。\n武田信玄  他 21 岁 时 流放 自己 的 父亲 武田信虎  至骏河 , 避免 父亲 传位 给 弟弟 , 从而 登上 了 第 19 代家督 之位 。\n他 将 信 浓国 ( 现 长野县 ) 纳入 控制 范围 后 , 又 与 当时 的 豪强 今井氏 、 北条 氏 结成 三国 军事同盟 , 与 上 杉谦信 在 川 中岛 前后 展开 了 五次 大战 。\n武田信玄  勇于 进攻 。\n他 连续 攻打 邻国 , 扩大 自己 势力范围 , 可称 遇神 杀神 , 遇佛 杀佛 。\n他 不仅 流放 了 自己 的 父亲 , 连 自己 的 嫡子 武田义信 因 与 他 在 战略 方向 上 相左 , 也 被 他 幽禁 于 佛寺 , 随即 被迫 自杀 。\n武田信玄  虽然 是 战国 武将 中 的 最强者 , 但 他 的 弱点 是 年龄 。\n信玄比 织田信长 年长 13 岁 , 比上 杉谦信 年长 9 岁 。\n当信 玄年 届 五十 之 时 , 信长 和 谦信 犹 在 壮年 。\n上杉谦信 而且 , 武田信玄  虽 驰骋 天下 , 却 未率 军 进过 京都 , 而 织田信长 在 永禄 十一年 ( 1568 年 ) 就 以 拥立 第 15 代 将军 足利义 昭 为名 率兵 上洛 了 。\n所谓 \" 制 京都 者 得 天下 \" , 所以 , 想要 一统天下 , 武田信玄  的 时间 很 紧迫 。\n元龟 三年 ( 1572 年 ) , 武田信玄  与 室 町 幕府 第 15 代 将军 足利义 昭 、 本愿 寺 显如 , 以及 浅井 氏 、 朝仓氏 等 反 织田信长 实力 组成 联盟 , 编织 \" 反信长 包围圈 \" 。\n同年 10 月 3 日 , 武田信玄  率领 大军 , 开始 了 第一次 上洛之行 。\n是 年 , 信玄 52 岁 , 这 也许 是 他 统一天下 的 最后 一次 机会 。\n武田信玄 所 率领 的 是 当时 战国 最强 的 3 万甲州 精兵 。\n打着 \" 风林火山 \" 的 旗帜 , 武田军 第一站 就 到达 了 织田信长 的 同盟 德川家康  所在 的 三河 远江 。\n织田信长 德川家康  的 军队 在 甲州 精兵 之前 显得 不堪一击 , 到 了 10 月 13 日 , 只来 成 、 天 方城 、 一 宫城 、 饭田 城 、 各和城 、 向 笠 城 等 城池 纷纷 被 攻陷 。\n德川家康  见势不妙 , 决定 在 浜松 城中 闭门不出 。\n但是 武田信玄  毫不 松懈 , 又 将 家康 在 远江 地区 的 重要 据点 二俣城 攻破 。\n德川家康  集合 所有 军队 共 1 万 1 千人 , 出城 与 信玄 决一死战 , 但 大败 而 还 , 险些 失 了 性命 。\n这次 战争 被 称为 \" 三方 原战 \" , 德川家康  曾经 承认 这次 战争 是 他 生平 最大 的 失败 。\n"}
```

## 中文预训练数据制作

下面是针对训练任务的数据集应用。

* llama为例

注：若使用llama模型，则不需要提前进行分词，请将WuDaoCorpus2.0_base_200G中的json文件预处理为如下格式的jsonl文件：
```
{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}
{"text": "PaddleNLP是自然语言..."}
```

之后利用如下脚本将对应的jsonl文件转化为.bin & .idx文件。
```shell
python -u  create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --tokenizer_name "LlamaTokenizer" \
    --input_path "wudao_corpus_200g.jsonl" \
    --output_prefix "wudao_corpus_200g" \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 10000 \
    --workers 48
```

* ernie为例
```shell
python -u  create_pretraining_data.py \
    --model_name "ernie-3.0-base-zh" \
    --tokenizer_name "ErnieTokenizer" \
    --input_path "wudao_corpus_200g.jsonl" \
    --output_prefix "wudao_corpus_200g"  \
    --data_format "JSON" \
    --json_key "text" \
    --split_sentences \
    --data_impl "mmap" \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func "jieba" \
    --cn_splited \
    --log_interval 10000 \
    --workers 48
```


- 我们提前进行了分词，所以加上了 `cn_splited`，否则不需要使用此选项。
- model_name 可以更换为[其他模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm)。
- workers 表示转化的线程数目

在当前目录下产出训练所需数据。
```
wudao_corpus_200g.bin
wudao_corpus_200g.idx
```
用户可以使用此数据进行预训练任务。
