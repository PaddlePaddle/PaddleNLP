# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2019 NVIDIA CORPORATION.
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

import bz2
import glob
import os
import requests
import subprocess

from opencc import OpenCC
from paddlenlp.utils.log import logger
from tqdm import tqdm


class WikicorpusTextFormatter:
    def __init__(self, language, save_path):
        assert language.lower() in ["en", "zh"], \
            'WikicorpusTextFormatting is not implemented for language %s yet.' % language

        self.language = language.lower()
        self.download_urls = {
            'en':
            'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2',
            'zh':
            'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2'
        }
        self.downloaded_files = {
            'en': 'wikicorpus_en.xml.bz2',
            'zh': 'wikicorpus_zh.xml.bz2'
        }
        self.chinese_coneverter = OpenCC('t2s')
        self.save_path = save_path + '/wikicorpus_' + language
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.formatted_file = os.path.join(self.save_path, "wiki_formatted.txt")

        self.download()
        self.merge()

    def download(self):
        url = self.download_urls[self.language]
        filename = self.downloaded_files[self.language]
        self.unzipped_file = os.path.join(self.save_path,
                                          filename.replace(".bz2", ""))

        if os.path.isfile(self.unzipped_file):
            logger.info(
                f'File {self.unzipped_file} already exists, skipping download.')
        else:
            response = requests.get(url=url, stream=True)
            content_size = int(response.headers['Content-Length']) / (1024 *
                                                                      1024)
            downloaded_file = os.path.join(self.save_path, filename)
            logger.info(
                f"Downloading: {url}, content size: {content_size:.2f} M and saving file to {downloaded_file}."
            )
            with open(downloaded_file, "wb") as handle:
                for data in tqdm(
                        iterable=response.iter_content(1024 * 1024),
                        total=content_size,
                        unit='M'):
                    handle.write(data)

            logger.info(f'Unzipping: {downloaded_file}')
            with open(self.unzipped_file, "wb") as new_file, bz2.BZ2File(
                    downloaded_file, "rb") as handle:
                for data in iter(lambda: handle.read(1024 * 1024), b""):
                    new_file.write(data)

        # Always do wikiextractor since this is relatively fast and will overwrite
        self.extracted_files = os.path.join(self.save_path, "extracted")
        if not os.path.exists(self.extracted_files):
            os.makedirs(self.extracted_files)
        subprocess.run('wikiextractor ' + self.unzipped_file + ' -o ' +
                       self.extracted_files,
                       shell=True,
                       check=True)

    def merge(self):
        # This puts one article per line
        logger.info(
            "Formatting the raw wiki texts and it takes some time to process. Please wait some minutes."
        )
        with open(self.formatted_file, mode='w', newline='\n') as ofile:
            for dirname in glob.glob(
                    self.extracted_files + '/*/', recursive=False):
                for filename in glob.glob(dirname + 'wiki_*', recursive=True):
                    logger.info(f"Formatting file {filename} .")
                    article_lines = []
                    article_open = False

                    with open(filename, mode='r', newline='\n') as file:
                        for line in file:
                            if self.language == "zh":
                                line = self.chinese_coneverter.convert(line)
                            if '<doc id=' in line:
                                article_open = True
                            elif '</doc>' in line:
                                article_open = False
                                for oline in article_lines[1:]:
                                    if oline != '\n':
                                        ofile.write(oline.rstrip() + " ")
                                ofile.write("\n\n")
                                article_lines = []
                            else:
                                if article_open:
                                    article_lines.append(line)


if __name__ == "__main__":
    wiki_formatting = WikicorpusTextFormatter("en", "./test")
