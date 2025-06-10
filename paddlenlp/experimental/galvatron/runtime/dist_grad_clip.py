from paddle.nn.clip import ClipGradBase
import paddle
import paddle.autograd as imperative_base
import paddle.distributed as dist
from paddle import _C_ops
from paddle.base import core, framework, unique_name
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.libpaddle import DataType
from paddle.common_ops_import import Variable, check_type, default_main_program
from paddle.distributed.utils.moe_utils import get_complete_pp_mesh
from paddle.framework import (
    LayerHelper,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from paddle.nn.clip import merge_selected_rows, get_tensor_from_selected_rows, _squared_l2_norm, _can_inplace_clip_grad

class ClipGradByGlobalNorm(ClipGradBase):
    r"""
    Given a list of Tensor :math:`t\_list` , calculate the global norm for the elements of all tensors in
    :math:`t\_list` , and limit it to ``clip_norm`` .

    - If the global norm is greater than ``clip_norm`` , all elements of :math:`t\_list` will be compressed by a ratio.

    - If the global norm is less than or equal to ``clip_norm`` , nothing will be done.

    The list of Tensor :math:`t\_list` is not passed from this class, but the gradients of all parameters set in ``optimizer``.
    If ``need_clip`` of specific param is ``False`` in its ``ParamAttr``, then the gradients of this param will not be clipped.

    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer``
    (for example: :ref:`api_paddle_optimizer_SGD`).

    The clipping formula is:

    .. math::

        t\_list[i] = t\_list[i] * \frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    Note:
        ``need_clip`` of ``ClipGradyGlobalNorm`` HAS BEEN DEPRECATED since 2.0.
        Please use ``need_clip`` in ``ParamAttr`` to specify the clip scope.

    Args:
        clip_norm (float): The maximum norm value.
        group_name (str, optional): The group name for this clip. Default value is ``default_group``.
        auto_skip_clip (bool, optional): skip clipping gradient. Default value is ``False``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10,
            ...                           weight_attr=paddle.ParamAttr(need_clip=True),
            ...                           bias_attr=paddle.ParamAttr(need_clip=False))
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            >>> sdg.step()
    """

    clip_norm: float
    group_name: str
    auto_skip_clip: bool

    def __init__(
        self,
        clip_norm: float,
        group_name: str = "default_group",
        auto_skip_clip: bool = False,
    ) -> None:
        super().__init__()
        self.clip_norm = float(clip_norm)
        self.group_name = group_name
        assert isinstance(auto_skip_clip, bool)
        self.auto_skip_clip = auto_skip_clip
        # TODO(zhiqiu): Now, in dygraph mode async_add_n is always used.
        # However, in static mode, it is only used in auto_parallel mode
        # by setting self._async_add_n to True. The reason is that there
        # are so many hard code depends on `add_n` in the legacy static
        # manual hybrid-parallel.
        self._async_add_n = None
        self.should_comm_on_shard_dim = False

    def __str__(self) -> str:
        return f"Gradient Clip By GlobalNorm, global_norm={self.clip_norm:f}"

    @imperative_base.no_grad()
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        if len(params_grads) > 0 and len(params_grads[0]) > 0:
            src_mesh = params_grads[0][0].process_mesh
        else:
            src_mesh = None
        
        print(f'[linguangming] [paddle.nn.clip] params_grads is {params_grads}, src_mesh is {src_mesh}')

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g

            if in_dynamic_mode() and g.is_selected_rows():
                merge_grad = merge_selected_rows(g)
                merge_grad = merge_grad._get_tensor_from_selected_rows()

            elif g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = merge_selected_rows(g)
                merge_grad = get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)

            # if the gradient mesh is not equal to src mesh
            # do reshard to get the result of squared_l2 from other pp stage mesh
            
            # linguangming 注释
            if src_mesh is not None and g.process_mesh != src_mesh:
                pp_mesh = get_complete_pp_mesh(g.process_mesh)
                if set(g.process_mesh.process_ids) < set(pp_mesh.process_ids):
                    sum_square = dist.reshard(
                        sum_square, pp_mesh, sum_square.placements
                    )
                print(f'[linguangming] [paddle.nn.clip] src_mesh is {src_mesh}, sum_square is {sum_square}')
                                
                sum_square = dist.reshard(
                    sum_square, src_mesh, sum_square.placements
                )
            # linguangming 注释

            if (
                sum_square.dtype == paddle.float16
                or sum_square.dtype == paddle.bfloat16
            ):
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == paddle.float32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

        # all parameters have been filtered out
        if (
            len(sum_square_list)
            + len(sum_square_list_fp16)
            + len(sum_square_list_fp32)
            == 0
        ):
            return params_grads

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"
        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_list_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_list)
            global_norm_var.append(global_norm_var_fp64)

        global_norm_var = async_add_n(global_norm_var)
        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(
            shape=[1], dtype=sum_dtype, fill_value=self.clip_norm
        )

        need_clip = False
        if not self.auto_skip_clip:  # always apply clip
            need_clip = True
            clip_var = paddle.divide(
                x=max_global_norm,
                y=paddle.maximum(x=global_norm_var, y=max_global_norm),
            )
        elif global_norm_var > max_global_norm:
            # only when global_norm_var > max_global_norm, grad need clip
            need_clip = True
            clip_var = paddle.divide(x=max_global_norm, y=global_norm_var)

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            if need_clip:
                clip_input = (
                    clip_var.astype(g.dtype)
                    if clip_var.dtype != g.dtype
                    else clip_var
                )
                if clip_input.process_mesh != g.process_mesh:
                    # TODO(pkuzyc): refine the reshard function between local
                    # and global mesh to avoid the following "_local_tensor()"
                    # operation.
                    if set(g.process_mesh.process_ids) < set(
                        clip_input.process_mesh.process_ids
                    ):
                        placements = clip_input.placements
                        is_replicate = True
                        for placement in placements:
                            if not placement.is_replicated():
                                is_replicate = False
                                break
                        if is_replicate:
                            clip_input = clip_input._local_value()
                        else:
                            raise NotImplementedError(
                                "Reshard a sharded tensor from a local mesh to a global mesh is not supported"
                            )
                    else:
                        pp_mesh = get_complete_pp_mesh(g.process_mesh)

                        if set(g.process_mesh.process_ids) < set(
                            pp_mesh.process_ids
                        ):
                            clip_input = dist.reshard(
                                clip_input, pp_mesh, clip_input.placements
                            )

                        clip_input = paddle.distributed.reshard(
                            clip_input, g.process_mesh, clip_input.placements
                        )

                if _can_inplace_clip_grad(g, clip_input):
                    g.multiply_(clip_input)
                    params_and_grads.append((p, g))
                else:
                    new_grad = paddle.multiply(g, clip_input)
                    params_and_grads.append((p, new_grad))
            else:
                params_and_grads.append((p, g))

        return params_and_grads

    def _pir_clip(self, params_grads):
        params_and_grads = []

        # no fusion grad
        no_fusion_sum_square = []
        no_fusion_sum_square_fp16 = []
        no_fusion_sum_square_fp32 = []

        # fusion grad need to commnuicate in dp&mp
        sum_square_dist = []
        sum_square_dist_fp16 = []
        sum_square_dist_fp32 = []

        # fusion grad only need to commnuicate in dp
        sum_square_not_dist = []
        sum_square_not_dist_fp16 = []
        sum_square_not_dist_fp32 = []

        auto_parallel_pp = False
        pp_meshes = set()
        pp_stage0_mesh = None
        for p, g in params_grads:
            if p.is_dist_dense_tensor_type():
                pp_meshes.add(p.dist_attr().process_mesh)
                if 0 in p.dist_attr().process_mesh.process_ids:
                    if pp_stage0_mesh is None:
                        pp_stage0_mesh = p.dist_attr().process_mesh
                    else:
                        p_mesh = p.dist_attr().process_mesh
                        if set(pp_stage0_mesh.process_ids) < set(
                            p_mesh.process_ids
                        ):
                            pp_stage0_mesh = p_mesh
                        assert set(p_mesh.process_ids) <= set(
                            pp_stage0_mesh.process_ids
                        )

        if len(pp_meshes) > 1:
            from paddle.distributed.auto_parallel.placement_type import (
                to_placements,
            )

            auto_parallel_pp = True
            assert pp_stage0_mesh is not None

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g

            if in_pir_mode() and g.is_selected_row_type():
                merge_grad = merge_selected_rows(g)
                merge_grad = get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)
            if (
                auto_parallel_pp
                and sum_square.dist_attr().process_mesh != pp_stage0_mesh
            ):
                sum_square = paddle.distributed.reshard(
                    sum_square,
                    pp_stage0_mesh,
                    to_placements(
                        sum_square.dist_attr().dims_mapping,
                        sum_square.dist_attr().process_mesh,
                        sum_square.dist_attr().partial_dims,
                    ),
                )
            if (
                not self.should_comm_on_shard_dim
                or p.optimize_attr["no_fusion"]
            ):
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    no_fusion_sum_square_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    no_fusion_sum_square_fp32.append(sum_square)
                else:
                    no_fusion_sum_square.append(sum_square)
            elif p.is_distributed:
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    sum_square_dist_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    sum_square_dist_fp32.append(sum_square)
                else:
                    sum_square_dist.append(sum_square)
            else:
                if (
                    sum_square.dtype == DataType.FLOAT16
                    or sum_square.dtype == DataType.BFLOAT16
                ):
                    sum_square_not_dist_fp16.append(sum_square)
                elif sum_square.dtype == DataType.FLOAT32:
                    sum_square_not_dist_fp32.append(sum_square)
                else:
                    sum_square_not_dist.append(sum_square)

        # all parameters have been filtered out
        if (
            len(no_fusion_sum_square)
            + len(no_fusion_sum_square_fp16)
            + len(no_fusion_sum_square_fp32)
            + len(sum_square_dist)
            + len(sum_square_dist_fp16)
            + len(sum_square_dist_fp32)
            + len(sum_square_not_dist)
            + len(sum_square_not_dist_fp16)
            + len(sum_square_not_dist_fp32)
            == 0
        ):
            return params_grads

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        sum_dtype = (
            'float64'
            if len(no_fusion_sum_square)
            + len(sum_square_dist)
            + len(sum_square_not_dist)
            > 0
            else "float32"
        )
        no_fusion_global_norm = []
        global_norm_dist = []
        global_norm_not_dist = []
        if len(no_fusion_sum_square_fp16) > 0:
            global_norm_var_fp16 = async_add_n(no_fusion_sum_square_fp16)
            no_fusion_global_norm.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_dist_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_dist_fp16)
            global_norm_dist.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_not_dist_fp16) > 0:
            global_norm_var_fp16 = async_add_n(sum_square_not_dist_fp16)
            global_norm_not_dist.append(global_norm_var_fp16.astype(sum_dtype))

        if len(no_fusion_sum_square_fp32) > 0:
            global_norm_var_fp32 = async_add_n(no_fusion_sum_square_fp32)
            if sum_dtype == 'float32':
                no_fusion_global_norm.append(global_norm_var_fp32)
            else:
                no_fusion_global_norm.append(
                    global_norm_var_fp32.astype(sum_dtype)
                )
        if len(sum_square_dist_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_dist_fp32)
            if sum_dtype == 'float32':
                global_norm_dist.append(global_norm_var_fp32)
            else:
                global_norm_dist.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_not_dist_fp32) > 0:
            global_norm_var_fp32 = async_add_n(sum_square_not_dist_fp32)
            if sum_dtype == 'float32':
                global_norm_not_dist.append(global_norm_var_fp32)
            else:
                global_norm_not_dist.append(
                    global_norm_var_fp32.astype(sum_dtype)
                )
        if len(no_fusion_sum_square) > 0:
            global_norm_var_fp64 = async_add_n(no_fusion_sum_square)
            no_fusion_global_norm.append(global_norm_var_fp64)
        if len(sum_square_dist) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_dist)
            global_norm_dist.append(global_norm_var_fp64)
        if len(sum_square_not_dist) > 0:
            global_norm_var_fp64 = async_add_n(sum_square_dist)
            global_norm_not_dist.append(global_norm_var_fp64)

        global_norm_var = None
        if len(no_fusion_global_norm) > 0:
            global_norm_var = async_add_n(no_fusion_global_norm)

        if len(global_norm_dist) > 0:
            global_norm_dist_var = async_add_n(global_norm_dist)
        elif self.should_comm_on_shard_dim and self.has_dist_param:
            global_norm_dist_var = paddle.full(
                shape=[1], dtype=sum_dtype, fill_value=0.0
            )

        if self.should_comm_on_shard_dim and self.has_dist_param:
            global_norm_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_dist_var, self.sharding_group.id, True, False
            )
            global_norm_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_dist_var, self.mp_group.id, True, False
            )
            if global_norm_var is None:
                global_norm_var = global_norm_dist_var
            else:
                global_norm_var = global_norm_var + global_norm_dist_var

        if len(global_norm_not_dist) > 0:
            global_norm_not_dist_var = async_add_n(global_norm_not_dist)
        elif self.should_comm_on_shard_dim and self.has_not_dist_param:
            global_norm_not_dist_var = paddle.full(
                shape=[1], dtype=sum_dtype, fill_value=0.0
            )
        if self.should_comm_on_shard_dim and self.has_not_dist_param:
            global_norm_not_dist_var = paddle._C_ops.c_allreduce_sum(
                global_norm_not_dist_var, self.sharding_group.id, True, False
            )
            if global_norm_var is None:
                global_norm_var = global_norm_not_dist_var
            else:
                global_norm_var = global_norm_var + global_norm_not_dist_var

        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(
            shape=[1], dtype=global_norm_var.dtype, fill_value=self.clip_norm
        )

        need_clip = False
        if not self.auto_skip_clip:  # always apply clip
            need_clip = True
            clip_var = paddle.divide(
                x=max_global_norm,
                y=paddle.maximum(x=global_norm_var, y=max_global_norm),
            )
        elif global_norm_var > max_global_norm:
            # only when global_norm_var > max_global_norm, grad need clip
            need_clip = True
            clip_var = paddle.divide(x=max_global_norm, y=global_norm_var)

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            if need_clip:
                clip_input = (
                    clip_var.astype(g.dtype)
                    if clip_var.dtype != g.dtype
                    else clip_var
                )
                if (
                    auto_parallel_pp
                    and clip_input.dist_attr().process_mesh
                    != g.dist_attr().process_mesh
                ):
                    clip_input = paddle.distributed.reshard(
                        clip_input,
                        g.dist_attr().process_mesh,
                        to_placements(
                            clip_input.dist_attr().dims_mapping,
                            clip_input.dist_attr().process_mesh,
                            clip_input.dist_attr().partial_dims,
                        ),
                    )

                new_grad = paddle.multiply(g, clip_input)
                params_and_grads.append((p, new_grad))
            else:
                params_and_grads.append((p, g))

        return params_and_grads

    def _static_clip(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_bf16 = []
        sum_square_list_fp32 = []

        def _add_n(var_list):
            if self._async_add_n:
                return paddle.stack(var_list).sum()
            else:
                return paddle.add_n(var_list)

        with framework.name_scope('gradient_clip'):
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    continue
                merge_grad = g
                with p.block.program._optimized_guard([p, g]):
                    if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                        merge_grad = merge_selected_rows(g)
                        merge_grad = get_tensor_from_selected_rows(merge_grad)
                    sum_square = _squared_l2_norm(merge_grad)
                    if sum_square.dtype == core.VarDesc.VarType.FP16:
                        sum_square_list_fp16.append(sum_square)
                    elif sum_square.dtype == core.VarDesc.VarType.BF16:
                        sum_square_list_bf16.append(sum_square)
                    elif sum_square.dtype == core.VarDesc.VarType.FP32:
                        sum_square_list_fp32.append(sum_square)
                    else:
                        sum_square_list.append(sum_square)

            if len(sum_square_list_fp16) > 0 and len(sum_square_list_bf16) > 0:
                raise NotImplementedError(
                    'FP16 and BF16 are not supported at the same time.'
                )

            # all parameters have been filtered out
            if (
                len(sum_square_list)
                + len(sum_square_list_fp16)
                + len(sum_square_list_fp32)
                == 0
            ) and (
                len(sum_square_list)
                + len(sum_square_list_bf16)
                + len(sum_square_list_fp32)
                == 0
            ):
                return params_grads

            with p.block.program._optimized_guard([p, g]):
                sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"

                global_norm_var = []
                if len(sum_square_list_fp16) > 0:
                    global_norm_var_fp16 = _add_n(sum_square_list_fp16)
                    if (
                        sum_square_list_fp32
                        or sum_square_list
                        or not _allow_pure_fp16_global_norm_clip()
                    ):
                        global_norm_var.append(
                            global_norm_var_fp16.astype(sum_dtype)
                        )
                    else:
                        global_norm_var.append(global_norm_var_fp16)
                if len(sum_square_list_bf16) > 0:
                    global_norm_var_bf16 = _add_n(sum_square_list_bf16)
                    if (
                        sum_square_list_fp32
                        or sum_square_list
                        or not _allow_pure_bf16_global_norm_clip()
                    ):
                        global_norm_var.append(
                            global_norm_var_bf16.astype(sum_dtype)
                        )
                    else:
                        global_norm_var.append(global_norm_var_bf16)
                if len(sum_square_list_fp32) > 0:
                    global_norm_var_fp32 = _add_n(sum_square_list_fp32)
                    if sum_dtype == 'float32':
                        global_norm_var.append(global_norm_var_fp32)
                    else:
                        global_norm_var.append(
                            global_norm_var_fp32.astype(sum_dtype)
                        )
                if len(sum_square_list) > 0:
                    # fp64
                    global_norm_var_other_dtype = _add_n(sum_square_list)
                    global_norm_var.append(global_norm_var_other_dtype)

                global_norm_var = (
                    _add_n(global_norm_var)
                    if len(global_norm_var) > 1
                    else global_norm_var[0]
                )
                global_norm_var = paddle.sqrt(x=global_norm_var)
                max_global_norm = paddle.full(
                    shape=[1],
                    dtype=global_norm_var.dtype,
                    fill_value=self.clip_norm,
                )
                scale_var = paddle.divide(
                    x=max_global_norm,
                    y=paddle.maximum(x=max_global_norm, y=global_norm_var),
                )
            param_new_grad_name_dict = {}
            for p, g in params_grads:
                if g is None:
                    continue
                if getattr(p, 'need_clip', True) is False:
                    params_and_grads.append((p, g))
                    continue

                with p.block.program._optimized_guard([p, g]):
                    new_g = _cast_to_mp_type_if_enabled(g)
                    # inplace
                    if (
                        new_g.dtype == core.VarDesc.VarType.FP16
                        and scale_var.dtype != core.VarDesc.VarType.FP16
                    ):
                        scale_input = scale_var.astype('float16')
                    elif (
                        new_g.dtype == core.VarDesc.VarType.BF16
                        and scale_var.dtype != core.VarDesc.VarType.BF16
                    ):
                        scale_input = scale_var.astype('bfloat16')
                    else:
                        scale_input = scale_var
                    # NOTE(Yuang Liu): For pure dp with gradient merge, the p and g
                    # will be in different blocks with the gradient clip related ops.
                    # We need to handle the correct block, otherwise will encounter
                    # a 'NotFoundError' during compile time.
                    block = default_main_program().current_block()
                    block.append_op(
                        type='elementwise_mul',
                        inputs={'X': new_g, 'Y': scale_input},
                        outputs={'Out': new_g},
                    )
                    if new_g is not g:
                        block.append_op(
                            type='cast',
                            inputs={'X': new_g},
                            outputs={'Out': g},
                            attrs={
                                'in_dtype': new_g.dtype,
                                'out_dtype': g.dtype,
                            },
                        )

                param_new_grad_name_dict[p.name] = g.name
                params_and_grads.append((p, g))

        _correct_clip_op_role_var(params_and_grads, param_new_grad_name_dict)
        return params_and_grads

    def _process_context(self, context, param, grad):
        if self.group_name not in context:
            context[self.group_name] = []
            context[self.group_name + "_clip_value"] = self.clip_norm
            context[self.group_name + "_clip"] = paddle.full(
                shape=[1], dtype=grad.dtype, fill_value=self.clip_norm
            )
        else:
            if not self.clip_norm == context[self.group_name + "_clip_value"]:
                raise ValueError(
                    "All parameters' 'clip_norm' of a same group should be the same"
                )

        merge_grad = grad
        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            merge_grad = merge_selected_rows(grad)
            merge_grad = get_tensor_from_selected_rows(merge_grad)
        elif in_pir_mode() and grad.is_selected_row_type():
            merge_grad = merge_selected_rows(grad)
            merge_grad = get_tensor_from_selected_rows(merge_grad)

        local_norm_var = _squared_l2_norm(merge_grad)
        context[self.group_name].append(local_norm_var)

        self.context = context

    def _create_operators(self, param, grad):
        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        group_scale_name = self.group_name + "_scale"
        if group_scale_name not in self.context:
            group_norm_var = async_add_n(self.context[self.group_name])
            group_norm_var = paddle.sqrt(x=group_norm_var)
            clip_var = self.context[self.group_name + "_clip"]
            group_scale_var = paddle.divide(
                x=clip_var,
                y=paddle.maximum(x=clip_var, y=group_norm_var),
            )
            assert group_scale_var.shape == (1,)
            self.context[group_scale_name] = group_scale_var

        if in_pir_mode():
            grad = paddle.multiply(grad, self.context[group_scale_name])
            return param, grad

        # inplace
        param.block.append_op(
            type='elementwise_mul',
            inputs={'X': grad, 'Y': self.context[group_scale_name]},
            outputs={'Out': grad},
        )

        return param, grad

