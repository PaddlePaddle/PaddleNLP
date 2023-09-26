import paddle
from paddlenlp_ops import get_output
import numpy as np
from paddlenlp.transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)
tokenizer = AutoTokenizer.from_pretrained("facebook/llama-13b")
if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.unk_token
rank_id = 0
is_blocking = True
paddle.device.set_device("cpu")
output_tensor = paddle.full(shape=[128 + 2, 1], fill_value=2, dtype="int64")
while True:
    outputs = []
    while True:
        get_output(output_tensor, rank_id, is_blocking)
        if (output_tensor[0, 0] == -2):  # read none
            continue
        # print("bs: ", output_tensor[1, 0].numpy())
        bsz = output_tensor[1, 0].numpy()
        output_numpy = output_tensor[2 : bsz + 2].numpy()
        # print("output: ", output_numpy[:1])
        output_numpy[output_numpy == -1] = 2
        outputs.append(output_numpy)
        if (output_tensor[0, 0] == -1): break
    output = np.concatenate(outputs, axis=1).tolist()
    for seq in output:
        seq = tokenizer.decode(seq)
        print("output: ", seq)
    print("end")