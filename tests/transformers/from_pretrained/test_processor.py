import unittest
import os
from paddlenlp.transformers import AutoProcessor, CLIPProcessor
from paddlenlp.utils.log import logger
from tests.testing_utils import slow


class ProcessorLoadTester(unittest.TestCase):
    # @slow
    def test_clip_load(self):
        logger.info("Download model from PaddleNLP BOS")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)

        logger.info("Download model from local")
        clip_processor.save_pretrained("./paddlenlp-test-model/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("./paddlenlp-test-model/clip-vit-base-patch32")
        clip_processor = AutoProcessor.from_pretrained("./paddlenlp-test-model/clip-vit-base-patch32")
        logger.info("Download model from PaddleNLP BOS with subfolder")
        clip_processor = CLIPProcessor.from_pretrained("./paddlenlp-test-model/", subfolder="clip-vit-base-patch32")
        clip_processor = AutoProcessor.from_pretrained("./paddlenlp-test-model/", subfolder="clip-vit-base-patch32")

        logger.info("Download model from PaddleNLP BOS with subfolder")
        clip_processor = CLIPProcessor.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )
        clip_processor = AutoProcessor.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )


        logger.info("Download model from HF HUB")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=True)
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=True)


        logger.info("Download model from aistudio")
        clip_processor = CLIPProcessor.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)
        clip_processor = AutoProcessor.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)

        logger.info("Download model from aistudio with subfolder")
        clip_processor = CLIPProcessor.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
        )
        clip_processor = AutoProcessor.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_aistudio=True
        )


        logger.info("Download model from modelscope")
        os.environ['from_modelscope'] = 'True'
        clip_processor = CLIPProcessor.from_pretrained("xiaoguailin/clip-vit-large-patch14")
        clip_processor = AutoProcessor.from_pretrained("xiaoguailin/clip-vit-large-patch14")


test = ProcessorLoadTester()
test.test_clip_load()