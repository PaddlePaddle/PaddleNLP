import paddle
from paddlenlp.utils.log import logger


class BenchmarkBase(object):

    def __init__(self):
        self.num_batch = 0

    @staticmethod
    def add_args(args, parser):
        parser = parser.add_argument_group()

    def create_data_loader(self, args, **kwargs):
        raise NotImplementedError

    def build_model(self, args, **kwargs):
        raise NotImplementedError

    def forward(self, model, args, input_data=None, **kwargs):
        raise NotImplementedError

    def logger(self,
               args,
               step_id=None,
               pass_id=None,
               batch_id=None,
               loss=None,
               batch_cost=None,
               reader_cost=None,
               num_samples=None,
               ips=None,
               **kwargs):
        logger.info(
            "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
            % (step_id, args.epoch * self.num_batch, loss, reader_cost,
               batch_cost, num_samples, ips))
