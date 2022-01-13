## Usage

```python
import paddle
from paddlenlp.transformers import VisualBertForPreTraining, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = VisualBertForPreTraining.from_pretrained('chenkangyang/visualbert-nlvr2-pre')

inputs = tokenizer("Welcome to paddlenlp.", return_attention_mask=True)
inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])

visual_embeds = paddle.ones([100, 1024]).unsqueeze(0)
visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

return_dict = True
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
    "return_dict": return_dict
})

outputs = model(**inputs)

if not return_dict:
    loss = outputs[0]
    prediction_logits = outputs[1]
    seq_relationship_logits = outputs[2]
    hidden_states = outputs[3]
    attentions = outputs[4]
else:
    loss = outputs['loss']
    prediction_logits = outputs['prediction_logits']
    seq_relationship_logits = outputs['seq_relationship_logits']
    hidden_states = outputs['hidden_states']
    attentions = outputs['attentions']
```

## Weights source

https://huggingface.co/uclanlp/visualbert-nlvr2-pre
