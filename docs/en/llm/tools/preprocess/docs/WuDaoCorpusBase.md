# WuDaoCorpus2.0 Base Corpus

| Name | Text Type | Plain Text Size |
|-|-|-|
| WuDaoCorpus2.0 Base| Chinese | 200GB |

WuDaoCorpora is a large-scale Chinese corpus collected by WuDao. The total data volume is 3TB. The currently open-sourced portion is the WuDaoCorpus2.0 base dataset, with a size of 200GB.

## Data Retrieval

**1. Download and Decompress**

Users can log in via WeChat on the [official website](https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab) to directly download the data. The downloaded compressed data is approximately 64GB. Decompress using:
```
unrar x WuDaoCorpus2.0_base_200G.rar
```

**2. Corpus Word Segmentation**

Due to the large size of the WuDao dataset, word segmentation is time-consuming. We first perform corpus segmentation:
```shell
python words_segmentation.py \
    --input_path ./WuDaoCorpus2.0_base_200G \
    --workers 40  \
    --data_format wudao \
    --cn_seg_func seg \
    --output_path ./wudao_lac_cut \
```

Note: For pretraining requiring SOP (Sentence Order Predict) tasks, we apply simple rules for text segmentation during word segmentation. If the corpus contains only one sentence, it's recommended to disable SOP loss by setting `binary_head=False` during training.

**3. Convert to jsonl Format**

After text conversion, we use `../data_tools/trans_to_json.py` to convert to jsonl format (with completed word segmentation):
```shell
python ./trans_to_json.py  \
    --input_path ./wudao_lac_cut \
    --output_path wudao_corpus_200g.jsonl \
    --workers 40
```
This generates the file `wudao_corpus_200g.jsonl` in the current directory. The format is as follows:
```
{"text": "Host : As a circuitous path to save the nation, we have no alternative.\nJin Xin : Exams and scores are merely interim evaluation methods, not the ultimate goal. Just as the purpose of human life isn't to eat, but eating sustains us, the aim of learning isn't for exams or grades. Rather, it's to acquire knowledge that becomes intrinsic capabilities, enabling us to create and work more effectively in the future.\nHost : Special thanks to Mr. Jin for today's interview, which has provided me with new perspectives on the rationale behind one-on-one education and the reasons for its successful development.\nAfter the program, could you share your vision and aspirations for the future of one-on-one education?\nJin Xin : Personalized one-on-one education is still in its infancy. If divided into four stages, we're currently transitioning from the first to the second phase. At Xue Da, we aim to progress faster and go further in this regard.\nIn the future, personalized education will undoubtedly help students improve academically, foster better growth, and ultimately cultivate more socially and nationally valuable talents. This is our trinity of achievements: academic performance, personal growth, and talent development.\nThe faculty team for Xue Da's 1-on-1 education comprises outstanding subject teachers, exam guidance experts, psychological counseling specialists, and learning strategy mentors, supplemented by full-time homeroom teachers and academic supervisors, ensuring comprehensive and well-organized instruction.\nSome faculty members have years of experience teaching graduating classes, having participated multiple times in middle school exam proposition research and grading. They deeply understand the essence of secondary education and can guide students to master key exam knowledge points efficiently for rapid score improvement.\n■ For academically challenged students: Focus on foundational knowledge, helping them build confidence through basic competencies, thereby achieving growth through self-assurance; Emphasize problem-solving methodologies and thought processes for subjective questions to strengthen application of fundamental knowledge.\n■ For students needing to elevate their performance: Identify individual weaknesses, reinforce foundations, and prioritize improvement in underperforming areas.\n"}
{"text": "Takeda Shingen was a born military commander who expanded his domain from 850,000 koku to over 900,000 koku during his lifetime.\nAt age 21, Takeda Shingen exiled his father Takeda Nobutora to Suruga to prevent him from passing the succession to his younger brother, thus securing his position as the 19th family head.\nAfter bringing Shinano Province (present-day Nagano Prefecture) under his control, he formed a tripartite military alliance with powerful clans like the Imai and Hojo, engaging in five major battles with Uesugi Kenshin at Kawanakajima.\nTakeda Shingen was known for his aggressive expansion. He continuously attacked neighboring territories to expand his sphere of influence, earning a reputation as an unstoppable conqueror.\nNot only did he exile his own father, but when his legitimate son Takeda Yoshinobu disagreed with his strategic direction, Shingen confined him to a Buddhist temple, leading to Yoshinobu's subsequent forced suicide.\nThough Takeda Shingen was among the strongest warlords of the Sengoku period, his weakness was age. Shingen was 13 years older than Oda Nobunaga and 9 years senior to Uesugi Kenshin.\nBy the time Shingen reached fifty, both Nobunaga and Kenshin were still in their prime. Moreover, while Takeda Shingen dominated the battlefield, he never marched on Kyoto, whereas Oda Nobunaga had already entered the capital in 1568 under the pretext of supporting the 15th Shogun Ashikaga Yoshiaki.\nThe saying \"He who controls Kyoto rules the nation\" meant time was pressing for Shingen's unification ambitions.\nIn 1572, Takeda Shingen allied with Ashikaga Yoshiaki (the 15th Shogun), Hongan-ji Kennyo, as well as the Asai and Asakura clans to form an \"Anti-Nobunaga Coalition\", creating an encirclement against Oda Nobunaga.\nOn October 3rd of that year, Takeda Shingen, then 52, began his first Kyoto campaign with 30,000 elite Koshu troops - perhaps his final opportunity to unify Japan.\nFlying the \"Fūrinkazan\" banner, the Takeda forces first targeted Tokugawa Ieyasu's territories in Mikawa and Tōtōmi, allies of Oda Nobunaga.\nThe Tokugawa forces proved no match for the Koshu veterans. By October 13th, multiple castles including Ushikubo, Tenryū, Ichinomiya, Iida, and Mukasa had fallen.\nTokugawa Ieyasu, recognizing the peril, decided to hole up in Hamamatsu Castle. But Takeda Shingen pressed on, capturing Ieyasu's key stronghold, Futamata Castle, in Tōtōmi.\nTokugawa Ieyasu rallied 11,000 troops for a decisive battle but suffered a crushing defeat, narrowly escaping with his life. This engagement, known as the Battle of Mikatagahara, was later acknowledged by Ieyasu as the greatest defeat of his career.\n"}
```
## Chinese Pretraining Data Preparation

The following applies to dataset preparation for training tasks.

### Using LLaMA as an Example

Note: If using LLaMA models, pre-tokenization is not required. Convert JSON files from WuDaoCorpus2.0_base_200G into jsonl format as follows:
```
{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}
{"text": "PaddleNLP is natural language..."}
```

Then use the following script to convert jsonl files to .bin & .idx files:
```shell
python -u create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --input_path "wudao_corpus_200g.jsonl" \
    --output_prefix "wudao_corpus_200g" \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 10000 \
    --workers 48
```

### Using ERNIE as an Example
```shell
python -u create_pretraining_data.py \
    --model_name "ernie-3.0-base-zh" \
    --tokenizer_name "ErnieTokenizer" \
    --input_path "wudao_corpus_200g.jsonl" \
    --output_prefix "wudao_corpus_200g" \
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

Notes:
- We pre-tokenized the data, hence added `cn_splited`. Omit this option if not pre-tokenized.
- model_name can be replaced with [other models](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm).
- --workers indicates the number of conversion threads.

The following files will be generated in current directory:
```
wudao_corpus_200g.bin
wudao_corpus_200g.idx
```
Users can utilize this data for pretraining tasks.
