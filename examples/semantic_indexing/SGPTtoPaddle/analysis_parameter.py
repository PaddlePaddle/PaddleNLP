import torch


def printParameter(
        pytorch_checkpoint_path="./source/pytorch_model.bin"
):
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    for k, v in pytorch_state_dict.items():
        print(k)


if __name__ == '__main__':
    # # analysis GPTJ
    printParameter("./source/pytorch_model.bin")
    # analysis bert
    # printParameter("./pytorch_model.bin")
