import time
import paddle


class Recorder(object):

    def __init__(self, batch_size, model_name, mem_info=None):
        self.model_name = model_name
        self.precision = "fp32"
        self.batch_size = batch_size

        self.infer_time = 0
        self.samples = 0

        self.start = 0

    def tic(self):
        self.start = time.time()

    def toc(self, samples=1):
        self.infer_time += (time.time() - self.start) * 1000
        self.samples += samples

    def report(self):
        print("----------------------- Env info ------------------------")
        print("paddle_version: {}".format(paddle.__version__))
        print("----------------------- Model info ----------------------")
        print("model_name: {}".format(self.model_name))
        print("model_type: {}".format(self.precision))
        print("----------------------- Data info -----------------------")
        print("batch_size: {}".format(self.batch_size))
        print("num_of_samples: {}".format(self.samples))
        print("----------------------- Perf info -----------------------")
        print("average_latency(ms): {}".format(self.infer_time /
                                               (self.samples)))
        print("QPS: {}".format((self.samples) / (self.infer_time / 1000.0)))
