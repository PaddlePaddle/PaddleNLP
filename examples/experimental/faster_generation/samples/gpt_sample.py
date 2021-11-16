from paddlenlp.transformers import GPTLMHeadModel, GPTChineseTokenizer
import paddle

model_name = 'gpt-cpm-small-cn-distill'

tokenizer = GPTChineseTokenizer.from_pretrained(model_name)
model = GPTLMHeadModel.from_pretrained(model_name)

inputs = '花间一壶酒，独酌无相亲。举杯邀明月，'
print("Model input:", inputs)
inputs = tokenizer(inputs)["input_ids"]
inputs = paddle.to_tensor(inputs, dtype='int64').unsqueeze(0)

outputs, _ = model.generate(
    input_ids=inputs, max_length=10, decode_strategy='greedy_search')

result = tokenizer.convert_ids_to_string(outputs[0].numpy().tolist())
print(result)
# 对影成三人。
