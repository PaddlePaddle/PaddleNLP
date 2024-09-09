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


import json
import logging
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import wordcloud
from pipelines.nodes.base import BaseComponent

from paddlenlp.taskflow.utils import download_file

# make sure that the font 'SimHei' is installed in system
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

URLS = {
    "SimHei": [
        "https://paddlenlp.bj.bcebos.com/applications/sentiment_analysis/SimHei.ttf",
        "c9c9de86d3fa7c4af0d3f1269bb2dff2",
    ],
}


class VisualSentiment(object):
    """
    A tool class for visualing sentiment analysis results.
    """

    def __init__(self, font_path=None):
        self.font_path = font_path
        self.wc = wordcloud.WordCloud(font_path=font_path, background_color="white", width=800, height=400)
        plt.figure(figsize=(8, 6))

    def _plot_wordcloud(self, content_freq, save_path):
        """
        plot wordcloud image.
        Args:
            content_freq (dict): an content dict with frequency, the key is content and its value is frequency.
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
            content_freq (dict): an content dict with frequency, the key is content and its value is frequency.
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
            content_freq (dict): an content dict with frequency, the key is content and its value is frequency.
            save_path (str): path that the image is saved to.
            image_type (str): Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].
            with_line_chart (bool): Whether to plot line chart, only work when image_type is set be histogram.
            top_n (int): show top_n of frequency of contents, only work when image_type is set be histogram.
            plt_title (str): the title of image, only work when image_type is set be histogram.
            plt_xlabel (str): the 'x' axis label of image, only work when image_type is set be histogram.
            plt_ylabel (str): the 'y' axis label of image, only work when image_type is set be histogram.
        """

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

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
            print("aspect_freq is empty, please check it.")
            exit(0)

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

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
            print("opinion_freq is empty, please check it.")
            exit(0)

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

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
            print("aspect_opinion is empty, please check it.")
            exit(0)

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

        if sentiment not in ["all", "positive", "negative"]:
            print(
                "Only 'all', 'positive' and 'negative' are received for sentiment, that is, you should set be in [all, positive, negative]."
            )
            exit(0)

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
            print("aspect_sentiment is empty, please check it.")
            exit(0)

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

        if image_type == "wordcloud":
            new_aspect_opinion = {}
            for aspect in aspect_sentiment:
                for sentiment in aspect_sentiment[aspect]:
                    key = aspect + sentiment
                    new_aspect_opinion[key] = aspect_sentiment[aspect][sentiment]
            self._plot_wordcloud(new_aspect_opinion, save_path)
        else:
            if top_n != 0 and descend_aspects is None:
                print("You should input the param descend_aspects when top_n != 0.")
                exit(0)

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
            print("aspect_opinion is empty, please check it.")
            exit(0)

        if aspect not in aspect:
            print("{} not in aspect_opinion, please check it.")
            exit(0)

        if image_type not in ["wordcloud", "histogram"]:
            print("Only wordcloud and histogram are supported, that is, you should set be in [wordcloud, histogram].")
            exit(0)

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


class SentimentResult(object):
    """
    load and analyze result of sentiment analysis.
    """

    def __init__(self, file_path, sentiment_name="情感倾向[正向,负向,未提及]", opinion_name="观点词"):
        self.file_path = file_path
        self.sentiment_name = sentiment_name
        self.opinion_name = opinion_name
        self.examples = self._load_json_file(file_path)
        self.read_and_count_result(self.examples)

    def _load_json_file(self, file_path: str):
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                items.append(item)
        return items

    def read_and_count_result(self, examples):
        sentiment_name = self.sentiment_name
        opinion_name = self.opinion_name
        aspect_frequency = defaultdict(int)
        opinion_frequency = defaultdict(int)
        aspect_opinion_positives = defaultdict(int)
        aspect_opinion_negatives = defaultdict(int)

        aspect_sentiment = defaultdict(dict)
        aspect_opinion = defaultdict(dict)
        for example in examples:
            if not example:
                continue
            for aspect in example["评价维度"]:
                aspect_name = aspect["text"]
                if "relations" not in aspect:
                    continue
                if sentiment_name not in aspect["relations"] or opinion_name not in aspect["relations"]:
                    continue
                sentiment = aspect["relations"][sentiment_name][0]
                if sentiment["text"] == "未提及":
                    continue
                aspect_frequency[aspect_name] += 1
                if sentiment["text"] not in aspect_sentiment[aspect_name]:
                    aspect_sentiment[aspect_name][sentiment["text"]] = 1
                else:
                    aspect_sentiment[aspect_name][sentiment["text"]] += 1

                opinions = aspect["relations"][opinion_name]
                for opinion in opinions:
                    opinion_text = opinion["text"]
                    opinion_frequency[opinion_text] += 1
                    if opinion_text not in aspect_opinion[aspect_name]:
                        aspect_opinion[aspect_name][opinion_text] = 1
                    else:
                        aspect_opinion[aspect_name][opinion_text] += 1

                    aspect_opinion_text = aspect_name + opinion_text
                    if sentiment["text"] == "正向":
                        aspect_opinion_positives[aspect_opinion_text] += 1
                    else:
                        aspect_opinion_negatives[aspect_opinion_text] += 1

        aspect_freq_items = sorted(aspect_frequency.items(), key=lambda x: x[1], reverse=True)
        descend_aspects = [item[0] for item in aspect_freq_items]

        self.aspect_frequency = aspect_frequency
        self.opinion_frequency = opinion_frequency
        self.aspect_sentiment = aspect_sentiment
        self.aspect_opinion = aspect_opinion
        self.aspect_opinion_positives = aspect_opinion_positives
        self.aspect_opinion_negatives = aspect_opinion_negatives
        self.descend_aspects = descend_aspects


class SentaVisualization(BaseComponent):
    """
    Generate Images with the Result of Sentiment Analysis.
    """

    outgoing_edges = 1

    def __init__(self, font_name: str = "SimHei"):
        """
        Init Senta Visualization.
        :param font_name: The font name used to generate images.
        """
        base_name_suffix = os.path.basename(URLS[font_name][0]).split(".")[-1]
        file_name = font_name + "." + base_name_suffix
        save_dir = os.path.dirname(__file__)
        download_file(save_dir, file_name, URLS[font_name][0], URLS[font_name][1])
        font_path = os.path.join(save_dir, file_name)
        self.vs = VisualSentiment(font_path)

    def run(self, sr_save_path: str):
        """
        :Param sr_save_path: The file path of sentiment analysis fesults.
        """
        if not os.path.exists(sr_save_path):
            raise ValueError("The sentiment analysis file not exist: {}".format(sr_save_path))

        img_save_dir = os.path.dirname(sr_save_path)
        sr = SentimentResult(sr_save_path)

        # generate images
        img_dict = {}
        # aspect analysis
        save_path = os.path.join(img_save_dir, "aspect_wc.png")
        self.vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="wordcloud")
        img_dict["aspect_wc"] = save_path
        save_path = os.path.join(img_save_dir, "aspect_hist.png")
        self.vs.plot_aspect_with_frequency(sr.aspect_frequency, save_path, image_type="histogram")
        img_dict["aspect_hist"] = save_path
        # opinion analysis
        save_path = os.path.join(img_save_dir, "opinion_wc.png")
        self.vs.plot_opinion_with_frequency(sr.opinion_frequency, save_path, image_type="wordcloud")
        img_dict["opinion_wc"] = save_path
        save_path = os.path.join(img_save_dir, "opinion_hist.png")
        self.vs.plot_opinion_with_frequency(sr.opinion_frequency, save_path, image_type="histogram")
        img_dict["opinion_hist"] = save_path
        # aspect and opinion analysis
        save_path = os.path.join(img_save_dir, "aspect_opinion_wc.png")
        self.vs.plot_aspect_with_opinion(sr.aspect_opinion, save_path, image_type="wordcloud", sentiment="all")
        img_dict["aspect_opinion_wc"] = save_path
        save_path = os.path.join(img_save_dir, "aspect_opinion_hist.png")
        self.vs.plot_aspect_with_opinion(
            sr.aspect_opinion, save_path, image_type="histogram", sentiment="all", top_n=8
        )
        img_dict["aspect_opinion_hist"] = save_path
        # positive aspect and opinion analysis
        save_path = os.path.join(img_save_dir, "aspect_opinion_wc_pos.png")
        self.vs.plot_aspect_with_opinion(
            sr.aspect_opinion_positives, save_path, image_type="wordcloud", sentiment="positive"
        )
        img_dict["aspect_opinion_wc_pos"] = save_path
        save_path = os.path.join(img_save_dir, "aspect_opinion_hist_pos.png")
        self.vs.plot_aspect_with_opinion(
            sr.aspect_opinion_positives, save_path, image_type="histogram", sentiment="positive", top_n=8
        )
        img_dict["aspect_opinion_hist_pos"] = save_path
        # netative aspect and opinion analysis
        save_path = os.path.join(img_save_dir, "aspect_opinion_wc_neg.png")
        self.vs.plot_aspect_with_opinion(
            sr.aspect_opinion_negatives, save_path, image_type="wordcloud", sentiment="negative"
        )
        img_dict["aspect_opinion_wc_neg"] = save_path
        save_path = os.path.join(img_save_dir, "aspect_opinion_hist_neg.png")
        self.vs.plot_aspect_with_opinion(
            sr.aspect_opinion_negatives, save_path, image_type="histogram", sentiment="negative", top_n=8
        )
        img_dict["aspect_opinion_hist_neg"] = save_path
        # aspect and sentiment analysis
        save_path = os.path.join(img_save_dir, "aspect_sentiment_wc.png")
        self.vs.plot_aspect_with_sentiment(sr.aspect_sentiment, save_path, image_type="wordcloud")
        img_dict["aspect_sentiment_wc"] = save_path
        save_path = os.path.join(img_save_dir, "aspect_sentiment_hist.png")
        self.vs.plot_aspect_with_sentiment(
            sr.aspect_sentiment, save_path, image_type="histogram", top_n=15, descend_aspects=sr.descend_aspects
        )
        img_dict["aspect_sentiment_hist"] = save_path

        logger.info("The images of sentiment analysis has been saved to: {}".format(img_save_dir))
        output = {"img_dict": img_dict}
        return output, "output_1"
