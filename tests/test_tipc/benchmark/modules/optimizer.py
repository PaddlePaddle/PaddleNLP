import paddle
import paddle.nn as nn


class OptimizerBenchmarkBase(object):

    def __init__(self):
        pass

    @staticmethod
    def add_args(args, parser):
        raise NotImplementedError

    def build_optimizer(self, args, learning_rate, model, **kwargs):
        raise NotImplementedError


class SGDBenchmark(OptimizerBenchmarkBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--max_grad_norm',
                            type=float,
                            default=None,
                            help='Norm clip. ')

    def build_optimizer(self, args, learning_rate, model, **kwargs):
        if getattr(args, "max_grad_norm", None) is not None:
            grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
        else:
            grad_clip = None

        optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,
                                         parameters=model.parameters(),
                                         grad_clip=grad_clip)

        return optimizer


class AdamBenchmark(OptimizerBenchmarkBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--max_grad_norm',
                            type=float,
                            default=None,
                            help='Norm clip. ')

    def build_optimizer(self, args, learning_rate, model, **kwargs):
        if getattr(args, "max_grad_norm", None) is not None:
            grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
        else:
            grad_clip = None

        optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,
                                          parameters=model.parameters(),
                                          grad_clip=grad_clip)

        return optimizer


class AdamWBenchmark(OptimizerBenchmarkBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--beta1', type=float, default=0.9, help='. ')
        parser.add_argument('--beta2', type=float, default=0.999, help='. ')
        parser.add_argument('--epsilon', type=float, default=1e-8, help='. ')
        parser.add_argument('--max_grad_norm',
                            type=float,
                            default=None,
                            help='. ')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0,
                            help='. ')

    def build_optimizer(self, args, learning_rate, model, **kwargs):
        if getattr(args, "max_grad_norm", None) is not None:
            grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
        else:
            grad_clip = None

        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "layer_norm"])
        ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            parameters=model.parameters(),
            grad_clip=grad_clip,
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        return optimizer
