import paddle.distributed as dist

def sep_reshard_layer(input, split_axis, concat_axis):
    # do alltoall operation to reshard input from [Shard(concat_axis)] to [Shard[split_axis]]
    sep_axis = input.process_mesh.dim_names.index("sep")

    input_placements = input.placements
    if not isinstance(input_placements[sep_axis], dist.Shard):
        raise ValueError(
            f"Input placements for 'sep' axis should be Shard({split_axis}), but got {input_placements[sep_axis]}"
        )

    if input_placements[sep_axis].get_dim() != concat_axis:
        raise ValueError(
            f"Input placements for 'sep' axis should be Shard({concat_axis}), but got {input_placements[sep_axis]}"
        )

    input_placements[sep_axis] = dist.Shard(split_axis)

    out = dist.reshard(input, input.process_mesh, input_placements)
    return out

def get_colwise_placements(use_sep=False):
    # When use ulysees, we do not need to split parameters
    # When not use ulysees, we split parameters by column, which means tensor parallel
    if use_sep:
        return [dist.Replicate(), dist.Replicate()] # [dp, ulysees-sp]
    else:
        return [dist.Replicate(), dist.Shard(1)] # [dp, tp]
    
def get_rowwise_placements(use_sep=False):
    # When use ulysees, we do not need to split parameters
    # When not use ulysees, we split parameters by row, which means pipeline parallel
    if use_sep:
        return [dist.Replicate(), dist.Replicate()] # [dp, ulysees-sp]
    else:
        return [dist.Replicate(), dist.Shard(0)] # [dp, tp]