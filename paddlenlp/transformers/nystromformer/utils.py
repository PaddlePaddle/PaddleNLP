import paddle


def gelu_python(x):
    return x * 0.5 * (1.0 + paddle.erf(x / paddle.sqrt(2.0)))


def gelu_new(x):
    return 0.5 * x * (1.0 + paddle.tanh(
        paddle.sqrt(2.0 / 3.141592653589793) *
        (x + 0.044715 * paddle.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + paddle.tanh(x * 0.7978845608 *
                                        (1.0 + 0.044715 * x * x)))


def quick_gelu(x):
    return x * paddle.nn.functional.sigmoid(1.702 * x)


def linear_act(x):
    return x


ACT2FN = {
    "relu": paddle.nn.functional.relu,
    "silu": paddle.nn.functional.silu,
    "swish": paddle.nn.functional.silu,
    "gelu": paddle.nn.functional.gelu,
    "tanh": paddle.tanh,
    "gelu_python": gelu_python,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": paddle.nn.functional.mish,
    "linear": linear_act,
    "sigmoid": paddle.nn.functional.sigmoid,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim,
                              *input_tensors):
    assert len(
        input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}")
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}")
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(
            input_tensor.chunk(
                num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(
            forward_fn(*input_tensors_chunk)
            for input_tensors_chunk in zip(*input_tensors_chunks))
        return paddle.concat(output_chunks, axis=chunk_dim)
    return forward_fn(*input_tensors)


def get_extended_attention_mask(attention_mask, input_shape):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def trans_matrix(matrix):
    dim = matrix.ndim
    trans_list = [i for i in range(dim - 2)] + [dim - 1, dim - 2]
    return matrix.transpose(trans_list)


def update_metrics(logits, labels, metrics):
    for metric in metrics:
        metric.update(logits.argmax(axis=1), labels)


def get_f1_score(precision, recall):
    p, r = precision.accumulate(), recall.accumulate()
    return 2 * p * r / (p + r)
