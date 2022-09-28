import paddle

from paddlenlp.transformers import LinearDecayWithWarmup


class SchedulerBase(object):

    def __init__(self):
        pass

    @staticmethod
    def add_args(args, parser):
        raise NotImplementedError

    def build_scheculer(self, args):
        raise NotImplementedError


class LambdaDecayBenchmark(SchedulerBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--epoch_start_decay',
                            type=int,
                            default=6,
                            help='epoch_start_decay')
        parser.add_argument('--lr_decay',
                            type=float,
                            default=0.8,
                            help='lr_decay')

    def build_scheculer(self, args):
        lr_scheduler = paddle.optimizer.lr.LambdaDecay(
            learning_rate=args.learning_rate,
            lr_lambda=lambda x: args.lr_decay**max(
                x + 1 - args.epoch_start_decay, 0.0),
            verbose=True)

        return lr_scheduler


class LinearDecayWithWarmupBenchmark(SchedulerBase):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(args, parser):
        parser.add_argument('--warmup_steps',
                            type=int,
                            default=0,
                            help='Warmup steps. ')
        parser.add_argument('--warmup_proportion',
                            type=float,
                            default=0.1,
                            help='Warmup proportion. ')

    def build_scheculer(self, args):
        warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, args.max_steps,
                                             warmup)

        return lr_scheduler
