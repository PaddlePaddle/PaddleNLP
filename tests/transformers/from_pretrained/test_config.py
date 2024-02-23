import unittest
import os
from paddlenlp.transformers import AutoConfig, BertConfig
from tests.testing_utils import slow
from paddlenlp.utils.log import logger


class ConfigLoadTester(unittest.TestCase):

    
    def test_config_load(self):
        logger.info("Download Config from PaddleNLP from diffenent sources")
        # 会从build-in加载，不会执行下载
        bert_config = BertConfig.from_pretrained("bert-base-uncased", from_hf_hub=True)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased", from_bos=True)

        # 因为不在build-in列表中，所以会从aistudio下载
        bert_config = AutoConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
        
        # 从modelscope下载模型
        os.environ['from_modelscope'] = 'True'
        bert_config = AutoConfig.from_pretrained("sdfdsfe/bert-base-uncased")
        os.environ['from_modelscope'] = 'False'


        logger.info("Download config from local dir, file existed")
        # 将文件下载到本地
        bert_config.save_pretrained("./paddlenlp-test-config/bert-base-uncased")
        # 指定文件夹路径进行加载
        bert_config = BertConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased")


        logger.info("Download config from local dir with subfolder")
        # 测试本地subfolder存在时的情况
        bert_config = BertConfig.from_pretrained("./paddlenlp-test-config", subfolder="bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddlenlp-test-config", subfolder="bert-base-uncased")

        # 测试本地没有要加载的文件夹
        try:
            bert_config = BertConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased-2")
        except:
            logger.info("dir not existed")

        
        logger.info("Download config from local file, file existed")
        # 测试直接加载文件
        bert_config = BertConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased/config.json")

        # 测试欲加载文件不在本地
        try:
            bert_config = AutoConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased/model_config.json")
        except:
            logger.info("file not existed")

        
        logger.info("Download Config from PaddleNLP from cache")
        # 由于之前下载放置到了默认cache目录，所以会直接从cache加载
        bert_config = AutoConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased", from_hf_hub=True)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased", from_bos=True)
        os.environ['from_modelscope'] = 'True'
        bert_config = AutoConfig.from_pretrained("sdfdsfe/bert-base-uncased")
        os.environ['from_modelscope'] = 'False'
        

        logger.info("Download Bert Config from PaddleNLP from different sources with subfolder")
        # 测试从不同源头下载存在subfolder的情况，modelscope传入subfolder无效
        bert_config = BertConfig.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="tiny-bert", from_hf_hub=True
        )
        bert_config = AutoConfig.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="tiny-bert", from_bos=True
        )
        bert_config = AutoConfig.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="tiny-bert", from_aistudio=True
        )


test = ConfigLoadTester()
test.test_config_load()