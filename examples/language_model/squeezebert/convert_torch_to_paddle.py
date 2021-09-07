from collections import OrderedDict


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):
    import torch
    import paddle
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for idx, (k, v) in enumerate(pytorch_state_dict.items()):
        if k.startswith('transformer.'):
            k = k.replace('transformer.', '')
        if 'weight' in k and v.ndim == 2 and 'embedding' not in k:
            v = v.transpose(0, 1)
        paddle_state_dict[k] = v.data.numpy().astype('float32')
    paddle.save(paddle_state_dict, paddle_dump_path)


def generate_config(config_file):
    import json
    fout = open(config_file.replace('config.json', 'model_config.json'), 'w+')
    with open(config_file) as f:
        config = json.load(f)
        config.pop('model_type')
        json.dump(config, fout, ensure_ascii=False)


if __name__ == "__main__":
    import torch

    convert_pytorch_checkpoint_to_paddle('./models/squeezebert-uncased/pytorch_model.bin',
                                         './models/squeezebert-uncased/model_state.pdparams')
    convert_pytorch_checkpoint_to_paddle('./models/squeezebert-mnli-headless/pytorch_model.bin',
                                         './models/squeezebert-mnli-headless/model_state.pdparams')

    convert_pytorch_checkpoint_to_paddle('./models/squeezebert-mnli/pytorch_model.bin',
                                         './models/squeezebert-mnli/model_state.pdparams')

    generate_config('./models/squeezebert-uncased/config.json')
    generate_config('./models/squeezebert-mnli-headless/config.json')
    generate_config('./models/squeezebert-mnli/config.json')
