# Day10 任务型对话作业辅导

本教程旨在辅导同学如何完成 AI Studio课程课后作业——[必修 | 任务式对话系统
](https://aistudio.baidu.com/aistudio/education/objective/25711)。

## 1. 用于订票的机器人主要使用以下哪种对话技术？

    A. 开放域对话
    B. 任务式对话
    C. 问答型对话

    正确答案：B

    解析：订票属于完成特定任务，主要使用任务式对话

## 2. 以下哪个不是Pipeline型任务式对话中的主要任务？

    A. NLU
    B. DST
    C. NLG
    D. ASR

    正确答案：D

    解析：Pipeline型任务式对话包含NLU、DST、Policy、NLG四个部分。ASR语音识别不在其中。

## 3. 以下哪个不是典型NLU任务的主要识别目标

    A. 意图
    B. 词槽
    C. 动作

    正确答案：C

    解析：典型NLU任务一般识别用户输入的意图和词槽。

## 4. 以下哪个数据集不可以用来进行DST任务？

    A. ATIS
    B. MultiWOZ
    C. CrossWOZ

    正确答案：A

    解析：ATIS是一个NLU的数据集，MultiWOZ和CrossWOZ是DST的数据集。

## 5. 关于任务型对话系统，以下表述正确的是？

    A. Pipeline型任务式对话系统的可解释性较弱。
    B. 相对Pipeline型系统，纯端到端型任务式对话系统不存在错误的累积和传播
    C. 商业系统中，一般使用的是端到端型任务式对话系统。
    D. 端到端型任务式对话系统采用的都是Seq2Seq框架。

    正确答案：B

    解析：相对Pipeline型系统，由于纯端到端型系统是整体联合优化，避免了Pipeline型系统中每个模块产生错误的的累积和传播。
