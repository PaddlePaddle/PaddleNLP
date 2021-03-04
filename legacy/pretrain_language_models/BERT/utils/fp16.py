#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import paddle
import paddle.fluid as fluid


def append_cast_op(i, o, prog):
    """
    Append a cast op in a given Program to cast input `i` to data type `o.dtype`.
    Args:
        i (Variable): The input Variable.
        o (Variable): The output Variable.
        prog (Program): The Program to append cast op.
    """
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={"in_dtype": i.dtype,
               "out_dtype": o.dtype})


def copy_to_master_param(p, block):
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p


def apply_dynamic_loss_scaling(loss_scaling, master_params_grads,
                               incr_every_n_steps, decr_every_n_nan_or_inf,
                               incr_ratio, decr_ratio):
    _incr_every_n_steps = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=incr_every_n_steps)
    _decr_every_n_nan_or_inf = fluid.layers.fill_constant(
        shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)

    _num_good_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_good_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)
    _num_bad_steps = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("num_bad_steps"),
        shape=[1],
        value=0,
        dtype='int32',
        persistable=True)

    grads = [fluid.layers.reduce_sum(g) for [_, g] in master_params_grads]
    all_grads = fluid.layers.concat(grads)
    all_grads_sum = fluid.layers.reduce_sum(all_grads)
    is_overall_finite = fluid.layers.isfinite(all_grads_sum)

    update_loss_scaling(is_overall_finite, loss_scaling, _num_good_steps,
                        _num_bad_steps, _incr_every_n_steps,
                        _decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

    # apply_gradient append all ops in global block, thus we shouldn't
    # apply gradient in the switch branch.
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            pass
        with switch.default():
            for _, g in master_params_grads:
                fluid.layers.assign(fluid.layers.zeros_like(g), g)


def create_master_params_grads(params_grads, main_prog, startup_prog,
                               loss_scaling):
    master_params_grads = []
    for p, g in params_grads:
        with main_prog._optimized_guard([p, g]):
            # create master parameters
            master_param = copy_to_master_param(p, main_prog.global_block())
            startup_master_param = startup_prog.global_block()._clone_variable(
                master_param)
            startup_p = startup_prog.global_block().var(p.name)
            append_cast_op(startup_p, startup_master_param, startup_prog)
            # cast fp16 gradients to fp32 before apply gradients
            if g.name.find("layer_norm") > -1:
                scaled_g = g / loss_scaling
                master_params_grads.append([p, scaled_g])
                continue
            master_grad = fluid.layers.cast(g, "float32")
            master_grad = master_grad / loss_scaling
            master_params_grads.append([master_param, master_grad])

    return master_params_grads


def master_param_to_train_param(master_params_grads, params_grads, main_prog):
    for idx, m_p_g in enumerate(master_params_grads):
        train_p, _ = params_grads[idx]
        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
            if train_p.name.find("layer_norm") > -1:
                fluid.layers.assign(m_p_g[0], train_p)
            else:
                append_cast_op(m_p_g[0], train_p, main_prog)


def update_loss_scaling(is_overall_finite, prev_loss_scaling, num_good_steps,
                        num_bad_steps, incr_every_n_steps,
                        decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
    """
    Update loss scaling according to overall gradients. If all gradients is 
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio. 
    Otherwisw, loss scaling will decrease by decr_ratio after 
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.
    Args:
        is_overall_finite (Variable): A boolean variable indicates whether 
                                     all gradients are finite.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which 
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which 
                                  some gradients are infinite.
        incr_every_n_steps (Variable): A variable represents increasing loss 
                                       scaling every n consecutive steps with 
                                       finite gradients.
        decr_every_n_nan_or_inf (Variable): A variable represents decreasing 
                                            loss scaling every n accumulated 
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           loss scaling.
    """
    zero_steps = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    with fluid.layers.Switch() as switch:
        with switch.case(is_overall_finite):
            should_incr_loss_scaling = fluid.layers.less_than(
                incr_every_n_steps, num_good_steps + 1)
            with fluid.layers.Switch() as switch1:
                with switch1.case(should_incr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * incr_ratio
                    loss_scaling_is_finite = fluid.layers.isfinite(
                        new_loss_scaling)
                    with fluid.layers.Switch() as switch2:
                        with switch2.case(loss_scaling_is_finite):
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                        with switch2.default():
                            pass
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

                with switch1.default():
                    fluid.layers.increment(num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)

        with switch.default():
            should_decr_loss_scaling = fluid.layers.less_than(
                decr_every_n_nan_or_inf, num_bad_steps + 1)
            with fluid.layers.Switch() as switch3:
                with switch3.case(should_decr_loss_scaling):
                    new_loss_scaling = prev_loss_scaling * decr_ratio
                    static_loss_scaling = \
                        fluid.layers.fill_constant(shape=[1],
                                             dtype='float32',
                                             value=1.0)
                    less_than_one = fluid.layers.less_than(new_loss_scaling,
                                                           static_loss_scaling)
                    with fluid.layers.Switch() as switch4:
                        with switch4.case(less_than_one):
                            fluid.layers.assign(static_loss_scaling,
                                                prev_loss_scaling)
                        with switch4.default():
                            fluid.layers.assign(new_loss_scaling,
                                                prev_loss_scaling)
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.assign(zero_steps, num_bad_steps)
                with switch3.default():
                    fluid.layers.assign(zero_steps, num_good_steps)
                    fluid.layers.increment(num_bad_steps)
