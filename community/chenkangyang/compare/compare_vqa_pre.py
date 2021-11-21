text_masked = "The capital of France is {mask}."
text = "The capital of France is Paris."

torch_model_name = "uclanlp/visualbert-vqa-pre"
paddle_model_name = "visualbert-vqa-pre"

import numpy as np
import paddle
import torch
from paddlenlp.transformers import BertTokenizer as PDBertTokenizer
from paddlenlp.transformers import \
    VisualBertForPreTraining as PDVisualBertForPreTraining
from transformers import BertTokenizer as PTBertTokenizer
from transformers import VisualBertForPreTraining as PTVisualBertForPreTraining

torch_model = PTVisualBertForPreTraining.from_pretrained("../checkpoint/" +
                                                         torch_model_name)
torch_tokenizer = PTBertTokenizer.from_pretrained("bert-base-uncased")
torch_model.eval()

torch_inputs = torch_tokenizer(
    text_masked, return_tensors="pt", max_length=128, padding="max_length")
torch_visual_embeds = torch.ones([100, 2048]).unsqueeze(0)
torch_visual_token_type_ids = torch.ones(
    torch_visual_embeds.shape[:-1], dtype=torch.int64)
torch_visual_attention_mask = torch.ones(
    torch_visual_embeds.shape[:-1], dtype=torch.int64)
torch_inputs.update({
    "visual_embeds": torch_visual_embeds,
    "visual_token_type_ids": torch_visual_token_type_ids,
    "visual_attention_mask": torch_visual_attention_mask
})

max_length = torch_inputs["input_ids"].shape[-1] + torch_visual_embeds.shape[-2]
torch_labels = torch_tokenizer(
    text, return_tensors="pt", padding="max_length",
    max_length=max_length)["input_ids"]
torch_sentence_image_labels = torch.tensor(1).unsqueeze(0)  # Batch_size

with torch.no_grad():
    torch_outputs = torch_model(
        **torch_inputs,
        labels=torch_labels,
        sentence_image_labels=torch_sentence_image_labels)

torch_loss = torch_outputs.loss.cpu().detach().numpy()
torch_prediction_logits = torch_outputs.prediction_logits.cpu().detach().numpy()
torch_seq_relationship_logits = torch_outputs.seq_relationship_logits.cpu(
).detach().numpy()

print("torch_loss:{}".format(torch_loss))
print("torch_prediction_logits shape:{}".format(torch_prediction_logits.shape))
print("torch_prediction_logits:{}".format(torch_prediction_logits))
print("torch_seq_relationship_logits shape:{}".format(
    torch_seq_relationship_logits.shape))
print("torch_seq_relationship_logits:{}".format(torch_seq_relationship_logits))

# ========================================================================================================
paddle_model = PDVisualBertForPreTraining.from_pretrained(paddle_model_name)
paddle_tokenizer = PDBertTokenizer.from_pretrained("bert-base-uncased")
paddle_model.eval()

paddle_inputs = paddle_tokenizer(
    text_masked,
    max_seq_len=128,
    pad_to_max_seq_len=True,
    return_attention_mask=True)
paddle_inputs['input_ids'] = paddle.to_tensor([paddle_inputs['input_ids']])
paddle_inputs['token_type_ids'] = paddle.to_tensor(
    [paddle_inputs['token_type_ids']])
paddle_inputs['attention_mask'] = paddle.to_tensor(
    [paddle_inputs['attention_mask']])
paddle_visual_embeds = paddle.ones([100, 2048]).unsqueeze(0)
paddle_visual_token_type_ids = paddle.ones(
    paddle_visual_embeds.shape[:-1], dtype=paddle.int64)
paddle_visual_attention_mask = paddle.ones(
    paddle_visual_embeds.shape[:-1], dtype=paddle.int64)

return_dict = False
paddle_inputs.update({
    "visual_embeds": paddle_visual_embeds,
    "visual_token_type_ids": paddle_visual_token_type_ids,
    "visual_attention_mask": paddle_visual_attention_mask,
    "return_dict": return_dict
})

max_length = paddle_inputs["input_ids"].shape[-1] + paddle_visual_embeds.shape[
    -2]
paddle_labels = paddle.to_tensor(
    paddle_tokenizer(
        text, max_seq_len=max_length, pad_to_max_seq_len=True)["input_ids"])
paddle_sentence_image_labels = paddle.to_tensor(1).unsqueeze(0)  # Batch_size

with paddle.no_grad():
    paddle_outputs = paddle_model(
        **paddle_inputs,
        labels=paddle_labels,
        sentence_image_labels=paddle_sentence_image_labels)

if not return_dict:
    paddle_loss = paddle_outputs[0].cpu().detach().numpy()
    paddle_prediction_logits = paddle_outputs[1].cpu().detach().numpy()
    paddle_seq_relationship_logits = paddle_outputs[2].cpu().detach().numpy()
else:
    paddle_loss = paddle_outputs['loss'].cpu().detach().numpy()
    paddle_prediction_logits = paddle_outputs['prediction_logits'].cpu().detach(
    ).numpy()
    paddle_seq_relationship_logits = paddle_outputs[
        'seq_relationship_logits'].cpu().detach().numpy()

print("paddle_loss:{}".format(paddle_loss))
print("paddle_prediction_logits shape:{}".format(
    paddle_prediction_logits.shape))
print("paddle_prediction_logits:{}".format(paddle_prediction_logits))
print("paddle_seq_relationship_logits shape:{}".format(
    paddle_seq_relationship_logits.shape))
print("paddle_seq_relationship_logits:{}".format(
    paddle_seq_relationship_logits))

# ==============================================================================
assert torch_prediction_logits.shape == paddle_prediction_logits.shape, "the output logits should have the same shape, but got : {} and {} instead".format(
    paddle_prediction_logits.shape, paddle_prediction_logits.shape)
assert torch_seq_relationship_logits.shape == paddle_seq_relationship_logits.shape, "the output logits should have the same shape, but got : {} and {} instead".format(
    torch_seq_relationship_logits.shape, paddle_seq_relationship_logits.shape)
prediction_logits_diff = torch_prediction_logits - paddle_prediction_logits
seq_relationship_logits = torch_seq_relationship_logits - paddle_seq_relationship_logits

print("prediction_logits_diff", np.amax(abs(prediction_logits_diff)))
print("prediction_logits_diff_mean", abs(prediction_logits_diff).mean())
print("seq_relationship_logits_diff", np.amax(abs(seq_relationship_logits)))
