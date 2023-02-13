# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Get PaddleNLP develop model list && before merge pr """
import io
import os


def get_model_list():
    """
    get model list from
    <https://github.com/PaddlePaddle/PaddleNLP/model_zoo/>
    <https://github.com/PaddlePaddle/PaddleNLP/examples/>
    """

    CI_MODEL_LIST = [
        "waybill_ie",
        "msra_ner",
        "glue",
        "bert",
        "skep",
        "bigbird",
        "electra",
        "gpt",
        "ernie",
        "xlnet",
        "ofa",
        "albert",
        "squad",
        "tinybert",
        "lexical_analysis",
        "seq2seq",
        "pretrained_models",
        "word_embedding",
        "ernie-ctm",
        "distilbert",
        "stacl",
        "transformer",
        "simbert",
        "ernie-doc",
        "transformer-xl",
        "ernie-m",
        "plato-xl",
        "pointer_summarizer",
        "question_matching",
        "few_shot",
        "unimo-text",
        "ernie-csc",
        "nptag",
        "ofa",
        "transformer",
        "DuIE",
        "tcn",
        "word_embedding",
        "unified_transformer",
        "lic2021_baseline",
        "vae-seq2seq",
        "msra_ner",
        "simbert",
        "clue",
        "pet",
        "bert",
        "ernie-ctm",
        "DuReader-yesno",
        "nptag",
        "semantic_indexing",
        "seq2seq",
        "pointer_summarizer",
        "bigbird",
        "unimo-text",
        "minilmv2",
        "wordtag",
        "simcse",
        "ernie-gen",
        "distill_lstm",
        "DuReader-robust",
        "ernie_matching",
        "rnn",
        "ernie-1.0",
        "stacl",
        "erniesage",
        "DuEE",
        "efl",
        "doc",
        "couplet",
        "rnnlm",
        "pp-minilm",
        "dgu",
        "mpnet",
        "textcnn",
        "p-tuning",
        "SQuAD",
        "elmo",
        "plato-2",
        "pretrained_models",
        "sentiment_analysis",
        "ernie-health",
        "gpt-3",
    ]
    examples_second_list = ["model_interpretation", "semantic_indexing", "lexical_analysis", "word_embedding"]

    model_list = os.listdir("model_zoo")
    examples_list = os.listdir("examples/")
    app_list = os.listdir("applications/")

    # remove model_list README
    model_list.remove("README.md")
    examples_list.remove("README.md")
    model_list.extend(app_list)
    model_list.extend(examples_second_list)
    for examples_model_list in examples_list:
        if examples_model_list not in examples_second_list:
            examples_model = os.listdir("examples/" + examples_model_list)
            if "README.md" in examples_model:
                examples_model.remove("README.md")
            model_list.extend(examples_model)

    all_examples_dict = set(sorted(model_list))
    no_test_models = []

    # get model list not in CI/CE
    for full_model in all_examples_dict:
        if full_model not in CI_MODEL_LIST:
            no_test_models.append(full_model)

    # save model list for CI run_ci.sh
    with io.open("./scripts/regression/model_list.txt", "w", encoding="utf-8") as list:
        for all_model in all_examples_dict:
            list.write("{}\n".format(all_model))
        list.close()
    return all_examples_dict


if __name__ == "__main__":
    get_model_list()
