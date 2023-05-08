import unittest
from tempfile import TemporaryDirectory

from paddlenlp.taskflow import Taskflow

class TestText2TextGenerationTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.chatbot = Taskflow(
            task="text2text_generation",
            model="__internal_testing__/tiny-random-chatglm",
            dtype='float32',
        )

    def test_single(self):
        input_text = "您好？"
        output_text = self.chatbot(input_text)
        self.assertTrue(len(output_text) == 1)
        self.assertIsInstance(output_text['result'], list)

    def test_batch(self):
        input_text = ["你好",'你是谁？']
        output_text = self.chatbot(input_text)
        self.assertTrue(len(output_text['result']) == 2)
        self.assertIsInstance(output_text['result'], list)