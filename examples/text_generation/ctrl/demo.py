import paddle
from paddlenlp.transformers import CTRLLMHeadModel, CTRLTokenizer


class Demo:

    def __init__(self,
                 model_name_or_path="ctrl",
                 max_predict_len=128,
                 repetition_penalty=1.2):
        self.tokenizer = CTRLTokenizer.from_pretrained(model_name_or_path)
        print("Loading the model parameters, please wait...")
        self.model = CTRLLMHeadModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_predict_len = max_predict_len
        self.repetition_penalty = repetition_penalty
        print("Model loaded.")

    # prediction function
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
    demo = Demo(model_name_or_path="ctrl",
                max_predict_len=128,
                repetition_penalty=1.2)
    input_text_list = [
        "Diet English : I lost 10 kgs! ; German : ", "Reviews Rating: 5.0",
        "Questions Q: What is the capital of India?",
        "Books Weary with toil, I haste me to my bed,"
    ]
    for text in input_text_list:
        demo.generate(text)
