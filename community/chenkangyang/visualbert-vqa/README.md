## Usage

```python
import paddle
from paddlenlp.transformers import BertTokenizer, VisualBertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("visualbert-vqa", num_classes=3129)
model.eval()

inputs = tokenizer("Who is eating the apple?", return_attention_mask=True)
inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])
visual_embeds = paddle.ones([100,2048]).unsqueeze(0)
visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

return_dict = True
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
    "return_dict": return_dict
})

labels = paddle.nn.functional.one_hot(paddle.to_tensor(50), num_classes=3129).astype(paddle.float32) # Batch size 1, Num labels 3092

with paddle.no_grad():
    outputs = model(**inputs, labels = labels)

if not return_dict:
    loss = outputs[0]
    logits = outputs[1]
else:
    loss = outputs['loss']
    logits = outputs['logits']

```

## Weights source

https://huggingface.co/uclanlp/visualbert-vqa
