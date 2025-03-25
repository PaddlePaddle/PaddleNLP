# FlashMask Reward Model Training

## 3.1 RM Model Training

### Data Preparation

We support data formats where each line contains a JSON-formatted dictionary with the following fields:

- `src`: `str, List(str)`, user dialogue content.
- `tgt`: `str, List(str)`, system response content.
- `response`: `str, List(str)`, includes both chosen and rejected responses.
- `sort`: `List(int)`, sort values are used to distinguish between chosen and rejected responses in the response field (responses with smaller sort values are rejected, those with larger sort values are chosen).

Sample data:

```json
{
    "src": ["Hello"],
    "tgt": ["Hi! How can I help you today?"],
    "response": [
        "Good morning! What can I do for you?",
        "I'm busy right now, please leave me alone."
    ],
    "sort": [1, 2]
}
```
```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],
    "sort": [1, 0]
}
```
To facilitate testing, we also provide an ad generation dataset that can be used directly:

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

### FlashMask RM

```bash
# RM Launch Command Reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/rm/flashmask/run_reward.py ./config/llama/rm_flashmask_argument.json
```
