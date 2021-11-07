import paddle
import torch

paddle.set_device("cpu")

model_pair_list = [
            ["./uclanlp/visualbert-vqa/pytorch_model.bin", "./paddle_visualbert/visualbert-vqa/model_state.pdparams"],
            ["./uclanlp/visualbert-vqa-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-vqa-pre/model_state.pdparams"],
            ["./uclanlp/visualbert-vqa-coco-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-vqa-coco-pre/model_state.pdparams"],
            ["./uclanlp/visualbert-nlvr2/pytorch_model.bin", "./paddle_visualbert/visualbert-nlvr2/model_state.pdparams"],
            ["./uclanlp/visualbert-nlvr2-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-nlvr2-pre/model_state.pdparams"],
            ["./uclanlp/visualbert-nlvr2-coco-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-nlvr2-coco-pre/model_state.pdparams"],
            ["./uclanlp/visualbert-vcr/pytorch_model.bin", "./paddle_visualbert/visualbert-vcr/model_state.pdparams"],
            ["./uclanlp/visualbert-vcr-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-vcr-pre/model_state.pdparams"],
            ["./uclanlp/visualbert-vcr-coco-pre/pytorch_model.bin", "./paddle_visualbert/visualbert-vcr-coco-pre/model_state.pdparams"],
        ]

# State_dict's keys mapping: from torch to paddle
keys_dict = {
    # about embeddings
    "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",

    # about encoder layer
    'encoder.layer': 'encoder.layer.layers',
    'attention.self.query': 'self_attn.q_proj',
    'attention.self.key': 'self_attn.k_proj',
    'attention.self.value': 'self_attn.v_proj',
    'attention.output.dense': 'self_attn.out_proj',
    'attention.output.LayerNorm.weight': 'norm1.weight',
    'attention.output.LayerNorm.bias': 'norm1.bias',
    'intermediate.dense': 'linear1',
    'output.dense': 'linear2',
    'output.LayerNorm.weight': 'norm2.weight',
    'output.LayerNorm.bias': 'norm2.bias',

    # about cls predictions
    'cls.predictions.transform.dense': 'cls.predictions.transform',
    'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
    'cls.predictions.transform.LayerNorm.weight': 'cls.predictions.layer_norm.weight',
    'cls.predictions.transform.LayerNorm.bias': 'cls.predictions.layer_norm.bias',
    'cls.predictions.bias': 'cls.predictions.decoder_bias'
}

for model_pair in model_pair_list:
    torch_model_path, paddle_model_path = model_pair
    torch_state_dict = torch.load(torch_model_path)
    paddle_state_dict = {}

    for torch_key in torch_state_dict:
        paddle_key = torch_key
        for k in keys_dict:
            if k in paddle_key:
                paddle_key = paddle_key.replace(k, keys_dict[k])
        if ('linear' in paddle_key) or ('proj' in paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("cls.weight" in paddle_key) or ('pooler.dense.weight' in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(
                torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(
                torch_state_dict[torch_key].cpu().numpy())

        print("torch: ", torch_key, "\t", torch_state_dict[torch_key].shape)
        print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape,
            "\n")

    paddle.save(paddle_state_dict, paddle_model_path)
    print("Saved to: ", paddle_model_path)
    print("="*50)
