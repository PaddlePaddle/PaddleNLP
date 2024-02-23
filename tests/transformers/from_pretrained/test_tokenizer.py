import unittest
import os
from paddlenlp.transformers import (
    AutoTokenizer,
    T5Tokenizer,
)
from paddlenlp.utils.log import logger


class TokenizerLoadTester(unittest.TestCase):
    def test_tokenizer_load(self):
        logger.info("Download Config from PaddleNLP from diffenent sources")
        # 会从build-in加载，不会执行下载
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", from_hf_hub=True)
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-small", from_bos=True)

        # 因为不在build-in列表中，所以会从aistudio下载
        t5_tokenizer = AutoTokenizer.from_pretrained("aistudio/t5-small", from_aistudio=True)

        # 从modelscope下载tokenizer
        os.environ['from_modelscope'] = 'True'
        mengzi_t5_tokenizer = AutoTokenizer.from_pretrained("langboat/mengzi-t5-base")
        os.environ['from_modelscope'] = 'False'

        
        logger.info("Download config from local dir, file existed")
        # 将文件下载到本地
        t5_tokenizer.save_pretrained("./paddlenlp-test-model/t5-small")
        # 指定文件夹路径进行加载
        t5_tokenizer = T5Tokenizer.from_pretrained("./paddlenlp-test-model/t5-small")
        t5_tokenizer = AutoTokenizer.from_pretrained("./paddlenlp-test-model/t5-small")


        logger.info("Download config from local dir with subfolder")
        # 测试本地subfolder存在时的情况
        t5_tokenizer = T5Tokenizer.from_pretrained("./paddlenlp-test-model", subfolder="t5-small")
        t5_tokenizer = AutoTokenizer.from_pretrained("./paddlenlp-test-model", subfolder="t5-small")

        # 测试本地没有要加载的文件夹
        try:
            t5_tokenizer = T5Tokenizer.from_pretrained("./paddlenlp-test-model/t5-small-2")
        except:
            logger.info("dir not existed")

        
        logger.info("Download Config from PaddleNLP from cache")
        # 由于之前下载放置到了默认cache目录，所以会直接从cache加载
        t5_tokenizer = AutoTokenizer.from_pretrained("aistudio/t5-small", from_aistudio=True)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", from_hf_hub=True)
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-small", from_bos=True)
        os.environ['from_modelscope'] = 'True'
        mengzi_t5_tokenizer = AutoTokenizer.from_pretrained("langboat/mengzi-t5-base")
        os.environ['from_modelscope'] = 'False'

        
        logger.info("Download Bert Config from PaddleNLP from different sources with subfolder")
        # 测试从不同源头下载存在subfolder的情况
        t5_tokenizer = T5Tokenizer.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="t5-small", from_hf_hub=True
        )
        t5_tokenizer = AutoTokenizer.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="t5-small", from_bos=True
        )
        t5_tokenizer = AutoTokenizer.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="t5-small", from_aistudio=True
        )


test = TokenizerLoadTester()
test.test_tokenizer_load()