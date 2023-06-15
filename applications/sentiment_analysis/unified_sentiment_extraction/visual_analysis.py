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

import argparse
import os
import random
import shutil
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import wordcloud
from utils import load_json_file

from paddlenlp.taskflow.utils import download_file
from paddlenlp.utils.log import logger

# make sure that the font 'SimHei' is installed in system
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

URLS = {
    "SimHei": [
        "https://paddlenlp.bj.bcebos.com/applications/sentiment_analysis/SimHei.ttf",
        "c9c9de86d3fa7c4af0d3f1269bb2dff2",
    ],
}

PROMPT_ITEMS = {
    "aspect_prompt": "评价维度",
    "opinion_prompt": "观点词",
    "sentiment_prompt_prefix": "情感倾向",
    "separator": "##",
    "not_mentioned_option": "未提及",
    "positive_option": "正向",
    "negative_option": "负向",
}


class VisualSentiment(object):
    """
    A tool class for visualing sentiment analysis results.
    """

    def __init__(self, font_path=None):
        if font_path is not None:
            if not os.path.isfile(font_path):
                raise ValueError("The param font_path passed in may not be a file: {}".format(font_path))
            self.font_path = font_path
        else:
            default_name = "SimHei"
            save_dir = os.path.dirname(__file__)
            download_file(save_dir, default_name + ".ttf", URLS[default_name][0], URLS[default_name][1])
            self.font_path = os.path.join(save_dir, default_name + ".ttf")

        self.wc = wordcloud.WordCloud(font_path=self.font_path, background_color="white", width=800, height=400)
        plt.figure(figsize=(8, 6))

    def _plot_wordcloud(self, content_freq, save_path):
        """
        plot wordcloud image.

        Args:
            content_freq (dict): a content dict with frequency, the key is content and its value is frequency.
            save_path (str): path that the image is saved to.
        """

        text_list = []
        for item in content_freq:
            text_list.extend([item] * content_freq[item])
        random.shuffle(text_list)
        text = " ".join(text_list)

        self.wc.generate(text)
        self.wc.to_file(save_path)

    def _plot_histogram(
        self, content_freq, save_path, with_line_chart="true", top_n=15, plt_title="", plt_xlabel="", plt_ylabel=""
    ):
        """
        generate histogram image. one aspect corresponds to one bar.

        Args:
            content_freq (dict): a content dict with frequency, the key is content and its value is frequency.
            save_path (str): path that the image is saved to.
            with_line_chart (bool): Whether to plot line chart, only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of contents, only work when image_type is set be histogram.
            plt_title (str): the title of image, only work when image_type is set be histogram.
            plt_xlabel (str): the 'x' axis label of image, only work when image_type is set be histogram.
            plt_ylabel (str): the 'y' axis label of image, only work when image_type is set be histogram.
        """

        content_freq_items = content_freq.items()
        content_freq_items = sorted(content_freq_items, key=lambda x: x[1], reverse=True)
        content_freq_items = content_freq_items[:top_n]

        x_data = [item[0] for item in content_freq_items]
        y_data = [item[1] for item in content_freq_items]

        for i in range(len(x_data)):
            plt.bar(x_data[i], y_data[i])

        if with_line_chart:
            plt.plot(x_data, y_data, "-")
        plt.title(plt_title)

        plt.xlabel(plt_xlabel)
        plt.ylabel(plt_ylabel)
        plt.savefig(save_path)
        plt.close()

    def _plot_content_with_frequency(
        self,
        content_freq,
        save_path,
        image_type="wordcloud",
        with_line_chart="true",
        top_n=15,
        plt_title="",
        plt_xlabel="",
        plt_ylabel="",
    ):
        """
        generate image for specified content, such as aspect, opinion and so on.
        two types of images are supported: wordcloud and histogram.

        Args:
            content_freq (dict): a content dict with frequency, the key is content and its value is frequency.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of contents, only work when image_type is set be histogram.
            plt_title (str): the title of image, only work when image_type is set be histogram.
            plt_xlabel (str): the 'x' axis label of image, only work when image_type is set be histogram.
            plt_ylabel (str): the 'y' axis label of image, only work when image_type is set be histogram.
        """

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        if image_type == "wordcloud":
            self._plot_wordcloud(content_freq, save_path)
        else:
            self._plot_histogram(
                content_freq,
                save_path,
                with_line_chart=with_line_chart,
                top_n=top_n,
                plt_title=plt_title,
                plt_xlabel=plt_xlabel,
                plt_ylabel=plt_ylabel,
            )

    def plot_aspect_with_frequency(
        self, aspect_freq, save_path, image_type="wordcloud", with_line_chart="true", top_n=15
    ):
        """
        generate image for aspect, two types of images are supported: wordcloud and histogram.
        this method can help analyze which aspects of the product/service are more important to customers.

        Args:
            aspect_freq (dict): an aspect dict with frequency, the key is aspect and its value is frequency.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, Only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of apsects, Only work when image_type is set be histogram.
        """

        if not aspect_freq:
            raise ValueError("aspect_freq is empty, please check it.")

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        if image_type == "wordcloud":
            self._plot_content_with_frequency(aspect_freq, save_path, image_type=image_type)
        else:
            title = "The histogram of aspect/frequency"
            xlabel = "aspect"
            ylabel = "frequency"

            self._plot_content_with_frequency(
                aspect_freq,
                save_path,
                image_type=image_type,
                with_line_chart=with_line_chart,
                top_n=top_n,
                plt_title=title,
                plt_xlabel=xlabel,
                plt_ylabel=ylabel,
            )

    def plot_opinion_with_frequency(
        self, opinion_freq, save_path, image_type="wordcloud", with_line_chart="true", top_n=15
    ):
        """
        generate image for opinion, two types of images are supported: wordcloud and histogram.
        this method can help analyze the whole impression of the product/service.

        Args:
            opinion_freq (dict): an opinion dict with frequency, the key is opinion and its value is frequency.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, Only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of opinions, Only work when image_type is set be histogram.
        """

        if not opinion_freq:
            raise ValueError("opinion_freq is empty, please check it.")

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        if image_type == "wordcloud":
            self._plot_content_with_frequency(opinion_freq, save_path, image_type=image_type)
        else:
            title = "The histogram of opinion/frequency"
            xlabel = "opinion"
            ylabel = "frequency"

            self._plot_content_with_frequency(
                opinion_freq,
                save_path,
                image_type=image_type,
                with_line_chart=with_line_chart,
                top_n=top_n,
                plt_title=title,
                plt_xlabel=xlabel,
                plt_ylabel=ylabel,
            )

    def plot_aspect_with_opinion(
        self, aspect_opinion, save_path, sentiment="all", image_type="wordcloud", with_line_chart="true", top_n=15
    ):
        """
        generate image with aspect and opinion, that is, combining apsect with opinion to display the more specifical opinions of aspect.
        this method can help you at two aspects: 1. mining custom's overall impression of products/services; 2. analyzing the quality of some aspect and improve it further.

        Args:
            aspect_opinion (dict[dict] or dict): when sentiment set be "all", a expected dict containing aspect, opinion and its frequency, the key is aspect and its value is a dict containing the aspect's opinion and frequency. when sentiment set be "positive" or "netative", a expected dict containing aspect with opinion and frequency, the key is aspect with opinion and its value is frequency.
            aspect_sentiment (dict[dict]): a dict containing aspect, sentiment and its frequency, the key is aspect and its value is a dict containing the aspect's sentiment and frequency.
            save_path (str): path that the image is saved to.
            sentiment (str): analyzing aspect with sentiment, Only "all", "positive" and "negative" are received. "positive" only analyzes positive aspects with opinions, "negative" only analyzes negative aspects with opinions, and "all" analyzes all apsects.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, Only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of opinions, Only work when image_type is set be histogram.
        """

        if not aspect_opinion:
            raise ValueError("aspect_opinion is empty, please check it.")

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        if sentiment not in ["all", "positive", "negative"]:
            raise ValueError(
                "Only 'all', 'positive' and 'negative' are received for sentiment, that is, you should set be in [all, positive, negative]."
            )

        if sentiment == "all":
            new_aspect_opinion = {}

            for aspect in aspect_opinion:
                for opinion in aspect_opinion[aspect]:
                    key = aspect + opinion
                    new_aspect_opinion[key] = aspect_opinion[aspect][opinion]
            aspect_opinion = new_aspect_opinion

        if image_type == "wordcloud":
            self._plot_content_with_frequency(aspect_opinion, save_path, image_type=image_type)
        else:
            if sentiment == "all":
                title = "The histogram of aspect with opinion/frequency"
            else:
                title = "The histogram of {} aspect with opinion/frequency".format(sentiment)
            xlabel = "aspect with opinion"
            ylabel = "frequency"

            self._plot_content_with_frequency(
                aspect_opinion,
                save_path,
                image_type=image_type,
                with_line_chart=with_line_chart,
                top_n=top_n,
                plt_title=title,
                plt_xlabel=xlabel,
                plt_ylabel=ylabel,
            )

    def plot_aspect_with_sentiment(
        self, aspect_sentiment, save_path, image_type="wordcloud", top_n=0, descend_aspects=None
    ):
        """
        generate image with aspect and sentiment, that is, combining apsect and sentiment to display the sentiment of aspect.
        This method can help you more intuitively analyze customers' direct impressions of aspects of products/services.

        Args:
            aspect_sentiment (dict[dict]): a dict containing aspect, sentiment and its frequency, the key is aspect and its value is a dict containing the aspect's sentiment and frequency.
            descend_aspects (dict): an aspect list, sorted by frequency in reverse order.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            top_n (int): show top_n of frequency of opinions, Only work when image_type is set be histogram. if top_n set be 0, it will plot all aspects in histogram.
        """

        if not aspect_sentiment:
            raise ValueError("aspect_sentiment is empty, please check it.")

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        if image_type == "wordcloud":
            new_aspect_opinion = {}
            for aspect in aspect_sentiment:
                for sentiment in aspect_sentiment[aspect]:
                    key = aspect + sentiment
                    new_aspect_opinion[key] = aspect_sentiment[aspect][sentiment]
            self._plot_wordcloud(new_aspect_opinion, save_path)
        else:
            if top_n != 0 and descend_aspects is None:
                raise ValueError("You should input the param descend_aspects when top_n != 0.")

            if top_n != 0:
                keep_aspects = set(descend_aspects[:top_n])

            aspects = []
            positives = []
            negatives = []
            for aspect, sentiment in aspect_sentiment.items():
                if top_n != 0 and aspect not in keep_aspects:
                    continue
                aspects.append(aspect)
                if "正向" in sentiment:
                    positives.append(sentiment["正向"])
                else:
                    positives.append(0)
                if "负向" in sentiment:
                    negatives.append(sentiment["负向"])
                else:
                    negatives.append(0)

            total_width, n = 0.8, 2
            width = total_width / n
            x_pos = [item - (total_width - width) / 2 for item in range(len(aspects))]
            x_neg = [item + width for item in x_pos]

            plt.bar(x_pos, positives, width=width, label="positive")
            plt.bar(x_neg, negatives, width=width, label="negative")
            plt.title("The histogram of aspect/sentiment")
            plt.xlabel("aspect")
            plt.ylabel("sentiment frequency")
            plt.xticks(x_pos, aspects)
            plt.legend()
            plt.savefig(save_path)
            plt.close()

    def plot_opinion_with_aspect(
        self, aspect, aspect_opinion, save_path, image_type="wordcloud", with_line_chart=True, top_n=15
    ):
        """
        generate opinion image for given aspect. This method can help you analyzing opinions for given aspects.

        Args:
            aspect (str): The set of aspect to analyze.
            aspect_opinion (dict[dict] or dict): when sentiment set be "all", a expected dict containing aspect, opinion and its frequency, the key is aspect and its value is a dict containing the aspect's opinion and frequency. when sentiment set be "positive" or "netative", a expected dict containing aspect with opinion and frequency, the key is aspect with opinion and its value is frequency.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, Only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of opinions, Only work when image_type is set be histogram.
        """

        if not aspect_opinion:
            raise ValueError("aspect_opinion is empty, please check it.")

        if aspect not in aspect:
            raise ValueError("{} not in aspect_opinion, please check it.")

        if image_type not in ["wordcloud", "histogram"]:
            raise ValueError(
                "Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram]."
            )

        opinions = aspect_opinion[aspect]
        opinion_items = sorted(opinions.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            opinion_items = opinion_items[:top_n]

        opinion_freq = {k: v for k, v in opinion_items}

        if image_type == "wordcloud":
            self._plot_wordcloud(opinion_freq, save_path)
        else:
            title = "The opinion analysis for aspect [{}] ".format(aspect)
            xlabel = "opinion"
            ylabel = "frequency"
            self._plot_histogram(
                opinion_freq,
                save_path,
                with_line_chart=with_line_chart,
                top_n=top_n,
                plt_title=title,
                plt_xlabel=xlabel,
                plt_ylabel=ylabel,
            )

    def plot_sentence_sentiment(self, sentence_sentiment, save_path):
        """
        generate image for sentence sentiment, only histogram are supported.
        this method can help analyze the customers' whole impression for product/service.

        Args:
            sentence_sentiment (dict): an sentiment dict with frequency, the key is sentiment polarity and its value is frequency.
            save_path (str): path that the image is saved to.
        """

        if not sentence_sentiment:
            raise ValueError("sentence_sentiment is empty, please check it.")

        title = "The histogram of sentence sentiment"
        xlabel = "sentiment polarity"
        ylabel = "frequency"

        self._plot_histogram(
            sentence_sentiment, save_path, with_line_chart=False, plt_title=title, plt_xlabel=xlabel, plt_ylabel=ylabel
        )


class SentimentResult(object):
    """
    load and analyze result of sentiment analysis.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.sentiment_prompt = PROMPT_ITEMS["sentiment_prompt"]
        self.sentiment_prompt_prefix = PROMPT_ITEMS["sentiment_prompt_prefix"]
        self.options = PROMPT_ITEMS["options"]
        self.opinion_prompt = PROMPT_ITEMS["opinion_prompt"]
        self.aspect_prompt = PROMPT_ITEMS["aspect_prompt"]
        self.not_mentioned_option = PROMPT_ITEMS["not_mentioned_option"]
        self.positive_option = PROMPT_ITEMS["positive_option"]
        self.negative_option = PROMPT_ITEMS["negative_option"]
        self.prompts = set()
        # load the result of sentiment analysis
        self.results = self._load_sentiment_result(file_path)
        # define the parsing middle result for sentiment analysis
        self.aspect_frequency = defaultdict(int)
        self.opinion_frequency = defaultdict(int)
        self.aspect_sentiment = defaultdict(dict)
        self.aspect_opinion = defaultdict(dict)
        self.aspect_opinion_positives = defaultdict(int)
        self.aspect_opinion_negatives = defaultdict(int)
        self.descend_aspects = []
        self.sentence_sentiment = defaultdict(int)
        # start to parse sentiment result
        self.parse_sentiment_result(self.results)

    def _load_sentiment_result(self, file_path):
        return load_json_file(file_path)

    def _parse_aspect(self, aspect):
        aspect_name = aspect["text"]
        self.aspect_frequency[aspect_name] += 1
        if "relations" not in aspect:
            return

        sentiment_name = None
        if self.sentiment_prompt in aspect["relations"].keys():
            sentiment = aspect["relations"][self.sentiment_prompt][0]
            sentiment_name = sentiment["text"]
            if sentiment_name == self.not_mentioned_option:
                sentiment_name = None
                return
            if sentiment_name not in self.aspect_sentiment[aspect_name]:
                self.aspect_sentiment[aspect_name][sentiment_name] = 1
            else:
                self.aspect_sentiment[aspect_name][sentiment_name] += 1

        if self.opinion_prompt in aspect["relations"].keys():
            opinions = aspect["relations"][self.opinion_prompt]
            for opinion in opinions:
                opinion_name = opinion["text"]
                self.opinion_frequency[opinion_name] += 1
                if opinion_name not in self.aspect_opinion[aspect_name]:
                    self.aspect_opinion[aspect_name][opinion_name] = 1
                else:
                    self.aspect_opinion[aspect_name][opinion_name] += 1

                if sentiment_name is not None:
                    aspect_opinion_name = aspect_name + opinion_name
                    if sentiment_name == self.positive_option:
                        self.aspect_opinion_positives[aspect_opinion_name] += 1
                    else:
                        self.aspect_opinion_negatives[aspect_opinion_name] += 1

        self.prompts.update(aspect["relations"].keys())

    def _parse_opinion(self, opinion):
        opinion_name = opinion["text"]
        self.opinion_frequency[opinion_name] += 1

    def _parse_sentiment_polarity(self, sentiment):
        sentiment_name = sentiment["text"]
        self.sentence_sentiment[sentiment_name] += 1

    def parse_one_result(self, result):
        for key in result.keys():
            if key == self.aspect_prompt:
                for aspect in result[self.aspect_prompt]:
                    self._parse_aspect(aspect)
            elif key == self.opinion_prompt:
                for opinion in result[self.opinion_prompt]:
                    self._parse_opinion(opinion)
            elif key == self.sentiment_prompt:
                sentiment = result[self.sentiment_prompt][0]
                self._parse_sentiment_polarity(sentiment)
            else:
                raise ValueError(
                    "Unknown key {} for sentiment analysis, you can check it as follows: 1. whether the parameter task_type is right; 2. whether the sentiment prompt {} created by the parameter options matches with the prompt {} in the file of sentiment analysis results; 3. whether the aspect_prompt, opinion_prompt or sentiment prompt are right.".format(
                        key, self.sentiment_prompt, key
                    )
                )
            self.prompts.add(key)

    def parse_sentiment_result(self, results):
        for result in results:
            if not result:
                continue
            self.parse_one_result(result)
        # parse descend_aspects
        descend_aspects_items = sorted(self.aspect_frequency.items(), key=itemgetter(1), reverse=True)
        self.descend_aspects = [item[0] for item in descend_aspects_items]
        # check whether sentiment prompt is parsed correctly
        for prompt in self.prompts:
            if prompt.startswith(self.sentiment_prompt_prefix) and prompt != self.sentiment_prompt:
                logger.warning(
                    "The visual images related to sentiment ploarity cannot be generated. Because the sentiment prompt {} created by the opinions you input cannot be match with the one {} in the file of sentiment analysis result.".format(
                        self.sentiment_prompt, prompt
                    )
                )


def default_visual_analysis(args):
    # checking generating environment
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    # update sentiment prompt according to task type
    if args.options:
        PROMPT_ITEMS["options"] = args.options
    else:
        if args.task_type == "ext":
            PROMPT_ITEMS["options"] = [
                PROMPT_ITEMS["positive_option"],
                PROMPT_ITEMS["negative_option"],
                PROMPT_ITEMS["not_mentioned_option"],
            ]
        else:
            PROMPT_ITEMS["options"] = [PROMPT_ITEMS["positive_option"], PROMPT_ITEMS["negative_option"]]
    PROMPT_ITEMS["sentiment_prompt"] = PROMPT_ITEMS["sentiment_prompt_prefix"] + "[{}]".format(
        ",".join(PROMPT_ITEMS["options"])
    )

    # define sr to process the result of sentiment analysis
    logger.info("Trying to parse sentiment analysis result: {}".format(args.file_path))
    sr = SentimentResult(args.file_path)
    # define vs to visualize sentiment result
    vs = VisualSentiment(font_path=args.font_path)
    logger.info("Start to generate visual images of sentiment analysis for you.")
    # visualize aspect with frequency
    if args.task_type == "ext" and sr.aspect_frequency:
        save_path = os.path.join(args.save_dir, "aspect_wc.png")
        vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="wordcloud")
        save_path = os.path.join(args.save_dir, "aspect_hist.png")
        vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="histogram")
    # visualize opinion with frequency
    if args.task_type == "ext" and sr.opinion_frequency:
        save_path = os.path.join(args.save_dir, "opinion_wc.png")
        vs.plot_opinion_with_frequency(sr.opinion_frequency, save_path, image_type="wordcloud")
        save_path = os.path.join(args.save_dir, "opinion_hist.png")
        vs.plot_opinion_with_frequency(sr.opinion_frequency, save_path, image_type="histogram")
    # visualize aspect and opinion
    if args.task_type == "ext" and sr.aspect_opinion:
        save_path = os.path.join(args.save_dir, "aspect_opinion_wc.png")
        vs.plot_aspect_with_opinion(sr.aspect_opinion, save_path, image_type="wordcloud", sentiment="all")
        save_path = os.path.join(args.save_dir, "aspect_opinion_hist.png")
        vs.plot_aspect_with_opinion(sr.aspect_opinion, save_path, image_type="histogram", sentiment="all", top_n=8)
    # visualize positive aspect and opinion
    if args.task_type == "ext" and sr.aspect_opinion_positives:
        save_path = os.path.join(args.save_dir, "aspect_opinion_wc_pos.png")
        vs.plot_aspect_with_opinion(
            sr.aspect_opinion_positives, save_path, image_type="wordcloud", sentiment="positive"
        )
        save_path = os.path.join(args.save_dir, "aspect_opinion_hist_pos.png")
        vs.plot_aspect_with_opinion(
            sr.aspect_opinion_positives, save_path, image_type="histogram", sentiment="positive", top_n=8
        )
    # visualize negative aspect and opinion
    if args.task_type == "ext" and sr.aspect_opinion_negatives:
        save_path = os.path.join(args.save_dir, "aspect_opinion_wc_neg.png")
        vs.plot_aspect_with_opinion(
            sr.aspect_opinion_negatives, save_path, image_type="wordcloud", sentiment="negative"
        )
        save_path = os.path.join(args.save_dir, "aspect_opinion_hist_neg.png")
        vs.plot_aspect_with_opinion(
            sr.aspect_opinion_negatives, save_path, image_type="histogram", sentiment="negative", top_n=8
        )
    # visualize aspect and sentiment
    if args.task_type == "ext" and sr.aspect_sentiment:
        save_path = os.path.join(args.save_dir, "aspect_sentiment_wc.png")
        vs.plot_aspect_with_sentiment(sr.aspect_sentiment, save_path, image_type="wordcloud")
        save_path = os.path.join(args.save_dir, "aspect_sentiment_hist.png")
        vs.plot_aspect_with_sentiment(
            sr.aspect_sentiment, save_path, image_type="histogram", top_n=15, descend_aspects=sr.descend_aspects
        )
    # visualize sentiment polarity for sentence
    if args.task_type == "cls" and sr.sentence_sentiment:
        save_path = os.path.join(args.save_dir, "sentence_sentiment.png")
        vs.plot_sentence_sentiment(sr.sentence_sentiment, save_path)

    if not os.listdir(args.save_dir):
        logger.info(
            "Nothing generated for task {}, please check that you input the correct parameter task_type or the result of sentiment analysis.".format(
                args.task_type
            )
        )
    else:
        logger.info("Visual images for sentiment analysis has been saved to: {}".format(args.save_dir))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, type=str, help="The result path of sentiment analysis.")
    parser.add_argument("--save_dir", default="./images", type=str, help="The saving path of images.")
    parser.add_argument("--font_path", default=None, type=str, help="The font Path for showing Chinese in wordcloud.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str, help="Two task types [ext, cls] are supported, ext represents the aspect-based extraction task and cls represents the sentence-level classification task, defaults to ext.")
    parser.add_argument("--options", type=str, nargs="+", help="Used only for the classification task, the options for classification")

    args = parser.parse_args()
    # ypdf: enable

    default_visual_analysis(args)
