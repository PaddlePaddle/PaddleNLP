## Usage

```python
import paddle
from paddlenlp.transformers import BertTokenizer, VisualBertForMultipleChoice

text = "Welcome to use paddle paddle and paddlenlp!"
choice0 = "Use it."
choice1 = "Like it."

model = VisualBertForMultipleChoice.from_pretrained("visualbert-vcr", num_classes=1)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()

inputs = tokenizer.batch_encode(batch_text_or_text_pairs=[[text, text], [choice0, choice1]], max_seq_len=128, pad_to_max_seq_len=True, return_attention_mask=True)
input_ids_list = []
token_type_ids_list = []
attention_mask_list = []
inputs_dict = {}

for item in inputs: 
    input_ids_list.append(paddle.to_tensor(item['input_ids']).unsqueeze(0))
    token_type_ids_list.append(paddle.to_tensor(item['token_type_ids']).unsqueeze(0))
    attention_mask_list.append(paddle.to_tensor(item['attention_mask']).unsqueeze(0))

inputs_dict = {
    "input_ids": paddle.concat(input_ids_list, axis=0).unsqueeze(0),
    "token_type_ids": paddle.concat(token_type_ids_list, axis=0).unsqueeze(0),
    "attention_mask": paddle.concat(attention_mask_list, axis=0).unsqueeze(0)
} 

visual_embeds = paddle.ones([100,512]).unsqueeze(0)
visual_embeds = visual_embeds.expand(shape=[1, 2, *visual_embeds.shape])
visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

labels = paddle.to_tensor(0).unsqueeze(0) # choice0 is correct (according to Wikipedia ;)), batch size 1

return_dict = True
inputs_dict.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
    "return_dict": return_dict
})

with paddle.no_grad():
    outputs = model(**inputs_dict, labels=labels)

if not return_dict:
    loss = outputs[0]
    logits = outputs[1]
else:
    loss = outputs['loss']
    logits = outputs['logits']
```

## Weights source

https://huggingface.co/uclanlp/visualbert-vcr