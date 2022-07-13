from typing import Tuple, List
from logging import getLogger
import paddle
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer

logger = getLogger(__name__)


class Demo:

    def __init__(self,
                 model_name_or_path,
                 max_predict_len=128,
                 repetition_penalty=1.2):
        self.tokenizer = GPTTokenizer.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.model = OPTForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_predict_len = max_predict_len
        self.repetition_penalty = repetition_penalty
        logger.info("Model loaded.")

    @paddle.no_grad()
    def generate(self, inputs, max_predict_len=None, repetition_penalty=None):
        max_predict_len = max_predict_len if max_predict_len is not None else self.max_predict_len
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

        ids = self.tokenizer(inputs)["input_ids"]
        input_ids = paddle.to_tensor([ids], dtype="int64")
        max_length = max(self.max_predict_len - input_ids.shape[1], 20)
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            repetition_penalty=self.repetition_penalty)[0][0]
        decode_outputs = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(outputs.cpu()))

        print(f"input text: {inputs}")
        print(f"output text: {decode_outputs}")
        print("=" * 50)


if __name__ == "__main__":

    demo = Demo(model_name_or_path="facebook/opt-125m",
                max_predict_len=128,
                repetition_penalty=1.2)
    input_text_list = [
        "Hello, I am conscious and",
    ]
    for text in input_text_list:
        demo.generate(text)
