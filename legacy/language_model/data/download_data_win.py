import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

from test_utils import download_file_and_uncompress


def download_cityscapes_dataset(savepath, extrapath):
    url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


if __name__ == "__main__":
    download_cityscapes_dataset(LOCAL_PATH, LOCAL_PATH)
    print("Dataset download finish!")
