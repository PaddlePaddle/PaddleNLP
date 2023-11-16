import os
import sys
import numpy as np

import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from ppfleetx.data import build_dataloader
from ppfleetx.distributed.apis import env
from ppfleetx.models import build_module
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.utils import config
from ppfleetx.data import build_auto_dataset
# paddle.set_default_dtype("float64")

paddle.enable_static()

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    paddle.set_device(cfg["Global"]["device"])

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    amp_config = cfg.Engine.mix_precision
    amp_enable = amp_config["enable"]
    amp_dtype = amp_config.get("dtype", "float16")
    amp_level = amp_config.get("level", "O2")
    use_main_grad = amp_config.get("use_main_grad", False)
    scale_loss = amp_config["scale_loss"]
    custom_black_list = amp_config["custom_black_list"]
    custom_white_list = amp_config["custom_white_list"]
    amp_list = paddle.static.amp.CustomOpLists(custom_black_list=custom_black_list,custom_white_list=custom_white_list)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    model = module.model

    cfg.Optimizer.lr.update(
        {
            "epochs": cfg.Engine.num_train_epochs,
            "step_each_epoch": len(train_data_loader),
            "total_steps": cfg.Engine.max_steps,
        }
    )
    lr_scheduler = build_lr_scheduler(cfg.Optimizer.lr)
    optimizer = build_optimizer(cfg.Optimizer, model, lr_scheduler)

    if amp_enable:
        print("amp_enable: ", amp_enable)
        print("amp_level: ", amp_level)
        print("amp_dtype: ", amp_dtype)
        optimizer = paddle.static.amp.decorate(
                    optimizer,
                    amp_list,
                    level=amp_level,
                    dtype=amp_dtype,
                    init_loss_scaling=scale_loss,
                    use_dynamic_loss_scaling=True,
                    use_amp_guard=False,
                    use_promote=True)

    var_tokens = paddle.static.data("tokens", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_position_ids = paddle.static.data("position_ids", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_labels = paddle.static.data("labels", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="int64")
    var_loss_mask = paddle.static.data("loss_mask", shape=[cfg.Global.local_batch_size, cfg.Data.Train.dataset.max_seq_len], dtype="float32")

    preds = model(var_tokens, var_position_ids)
    loss = module.loss_fn(preds, var_labels, var_loss_mask)
    _, params_grads = optimizer.minimize(loss)

    exe = paddle.static.Executor(paddle.framework.CUDAPlace(paddle.distributed.ParallelEnv().dev_id))
    outs = exe.run(paddle.static.default_startup_program(), fetch_list=[p.name for p, g in params_grads])
    optimizer.amp_init(paddle.framework.CUDAPlace(paddle.distributed.ParallelEnv().dev_id), scope=paddle.static.global_scope())
    
    global_batch_size = cfg.Global.global_batch_size
    max_steps = cfg.Engine.max_steps
    # print(paddle.static.default_main_program())
    for step, batch in enumerate(train_data_loader()):
        tokens, position_ids, labels, loss_mask = batch

        outs = exe.run(
            paddle.static.default_main_program(), 
            feed={
                "tokens": tokens, 
                "position_ids": position_ids, 
                "labels": labels, 
                "loss_mask": loss_mask
            }, 
            fetch_list=[loss.name]
        )
        lr_scheduler.step(global_batch_size)

        print(
            "step: %d/%d\t" % (step, max_steps),
            "loss:%.9f\t" % outs[0],
            # "lr:%.5e\t" % optimizer.get_lr(),
        )
        if step >= max_steps:
            break
