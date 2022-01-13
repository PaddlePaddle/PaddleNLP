text = "Welcome to use paddle paddle and paddlenlp!"
choice0 = "Use it."
choice1 = "Like it."

torch_model_name = "uclanlp/visualbert-vcr"
paddle_model_name = "visualbert-vcr"

import numpy as np
import paddle
import torch
from paddlenlp.transformers import BertTokenizer as PDBertTokenizer
from paddlenlp.transformers import \
    VisualBertForMultipleChoice as PDVisualBertForMultipleChoice
from transformers import BertTokenizer as PTBertTokenizer
from transformers import \
    VisualBertForMultipleChoice as PTVisualBertForMultipleChoice

torch_model = PTVisualBertForMultipleChoice.from_pretrained("../checkpoint/" +
                                                            torch_model_name)
torch_tokenizer = PTBertTokenizer.from_pretrained("bert-base-uncased")
torch_model.eval()

torch_inputs = torch_tokenizer(
    [[text, text], [choice0, choice1]],
    return_tensors="pt",
    padding="max_length",
    max_length=128)
torch_inputs = {k: v.unsqueeze(0) for k, v in torch_inputs.items()}

torch_visual_embeds = torch.ones([100, 512]).unsqueeze(0)
torch_visual_embeds = torch_visual_embeds.expand(1, 2,
                                                 *torch_visual_embeds.shape)
torch_visual_token_type_ids = torch.ones(
    torch_visual_embeds.shape[:-1], dtype=torch.int64)
torch_visual_attention_mask = torch.ones(
    torch_visual_embeds.shape[:-1], dtype=torch.int64)

torch_inputs.update({
    "visual_embeds": torch_visual_embeds,
    "visual_token_type_ids": torch_visual_token_type_ids,
    "visual_attention_mask": torch_visual_attention_mask,
})

torch_labels = torch.tensor(0).unsqueeze(
    0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
with torch.no_grad():
    torch_outputs = torch_model(**torch_inputs, labels=torch_labels)

torch_loss = torch_outputs[0].cpu().detach().numpy()
torch_logits = torch_outputs[1]
torch_array = torch_logits.cpu().detach().numpy()

print("torch_prediction_loss:{}".format(torch_loss))
print("torch_prediction_logits shape:{}".format(torch_array.shape))
print("torch_prediction_logits:{}".format(torch_array))

# ========================================================================================================
paddle_model = PDVisualBertForMultipleChoice.from_pretrained(
    paddle_model_name, num_classes=1)
paddle_tokenizer = PDBertTokenizer.from_pretrained("bert-base-uncased")
paddle_model.eval()

paddle_inputs = paddle_tokenizer.batch_encode(
    batch_text_or_text_pairs=[[text, text], [choice0, choice1]],
    max_seq_len=128,
    pad_to_max_seq_len=True,
    return_attention_mask=True)
paddle_input_ids_list = []
paddle_token_type_ids_list = []
paddle_attention_mask_list = []
paddle_inputs_dict = {}

for item in paddle_inputs:
    paddle_input_ids_list.append(
        paddle.to_tensor(item['input_ids']).unsqueeze(0))
    paddle_token_type_ids_list.append(
        paddle.to_tensor(item['token_type_ids']).unsqueeze(0))
    paddle_attention_mask_list.append(
        paddle.to_tensor(item['attention_mask']).unsqueeze(0))

paddle_inputs_dict = {
    "input_ids": paddle.concat(
        paddle_input_ids_list, axis=0).unsqueeze(0),
    "token_type_ids": paddle.concat(
        paddle_token_type_ids_list, axis=0).unsqueeze(0),
    "attention_mask": paddle.concat(
        paddle_attention_mask_list, axis=0).unsqueeze(0)
}

paddle_visual_embeds = paddle.ones([100, 512]).unsqueeze(0)
paddle_visual_embeds = paddle_visual_embeds.expand(
    shape=[1, 2, *paddle_visual_embeds.shape])
paddle_visual_token_type_ids = paddle.ones(
    paddle_visual_embeds.shape[:-1], dtype=paddle.int64)
paddle_visual_attention_mask = paddle.ones(
    paddle_visual_embeds.shape[:-1], dtype=paddle.int64)

paddle_labels = paddle.to_tensor(0).unsqueeze(
    0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

return_dict = False
paddle_inputs_dict.update({
    "visual_embeds": paddle_visual_embeds,
    "visual_token_type_ids": paddle_visual_token_type_ids,
    "visual_attention_mask": paddle_visual_attention_mask,
    "return_dict": return_dict
})

with paddle.no_grad():
    paddle_outputs = paddle_model(**paddle_inputs_dict, labels=paddle_labels)

if not return_dict:
    paddle_loss = paddle_outputs[0].cpu().detach().numpy()
    paddle_logits = paddle_outputs[1]
else:
    paddle_loss = paddle_outputs['loss']
    paddle_logits = paddle_outputs['logits']
paddle_array = paddle_logits.cpu().detach().numpy()

print("paddle_prediction_loss:{}".format(paddle_loss))
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))

assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(
    torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))
