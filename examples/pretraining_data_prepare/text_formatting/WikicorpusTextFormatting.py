# Copyright (c) 2019 NVIDIA CORPORATION & 2021 PaddlePaddle Authors. 
# All rights reserved.
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
import urllib.request

from tqdm import tqdm


class WikicorpusTextFormatting:
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

        self.save_path = save_path + '/wikicorpus_' + language
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.formatted_file = os.path.join(self.save_path, "wiki_formatted.txt")

        self.download()
        self.merge()

    def download(self):
        url = self.download_urls[self.language]
        filename = self.downloaded_files[self.language]
        downloaded_file = os.path.join(self.save_path, filename)

        print('Downloading:', url)
        if os.path.isfile(self.save_path + '/' + filename):
            print(f'File {filename} already exists, skipping download.')
        else:
            response = urllib.request.urlopen(url)
            content_size = int(response.headers['Content-Length']) / 1024
            print(
                f"Downloading: {url}, content size: {content_size} k and saving file to {downloaded_file}."
            )
            with open(downloaded_file, "wb") as handle:
                for data in tqdm(
                        iterable=response.iter_content(1024),
                        total=content_size,
                        unit='k'):
                    handle.write(data)

            # Always unzipping since this is relatively fast and will overwrite
            print('Unzipping: ', downloaded_file)
            self.unzipped_file = os.path.join(self.save_path,
                                              filename.replace(".bz2", ""))
            file_size = os.path.getsize(self.unzipped_file) / 1024 * 1024
            with open(self.unzipped_file, "wb") as new_file, bz2.BZ2File(
                    downloaded_file, "rb") as f:
                for data in tqdm(
                        iterable=iter(lambda: f.read(1024 * 1024)),
                        total=file_size,
                        unit="m"):
                    new_file.write(data)
            os.removedirs(downloaded_file)

            # subprocess.run('bzip2 -dk ' + downloaded_file, shell=True, check=True)

            # for data in iter(lambda : f.read(1024 * 1024), ''):
            #     new_file.write(data)

    def merge(self):
        # This puts one article per line
        print(
            "Formatting the raw wiki texts and it takes some time to process. Please wait some minutes."
        )
        cnt = 0
        with open(self.formatted_file, mode='w', newline='\n') as ofile, \
            open(self.unzipped_file, mode='r', newline='\n') as f:
            for index, line in enumerate(f):
                if index % 10000:
                    print(f"Precoessing the line number {index} .")

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
    wiki_formatting = WikicorpusTextFormatting("zh", "./test")
