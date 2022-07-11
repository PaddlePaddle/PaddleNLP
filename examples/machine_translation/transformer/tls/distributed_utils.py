import paddle
import paddle.distributed as dist


def all_gather_tokens(data):
    """Gathers num of tokens from all nodes. 
       `data` should be a tensor of num of tokens.
    """
    if dist.get_world_size() < 2:
        return data
    if not hasattr(all_gather_tokens,
                   '_in_buffer') or all_gather_tokens._in_buffer is None:
        all_gather_tokens._in_buffer = data
        all_gather_tokens._out_buffers = []
    in_buffer = all_gather_tokens._in_buffer
    out_buffers = all_gather_tokens._out_buffers

    dist.all_gather(out_buffers, in_buffer)

    return paddle.add_n(out_buffers)
