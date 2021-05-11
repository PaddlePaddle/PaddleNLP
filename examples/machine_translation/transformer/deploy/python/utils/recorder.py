import os
import time
import paddle
import pynvml
import psutil
import GPUtil


class Recorder(object):
    def __init__(self, config, batch_size, model_name, mem_info=None):
        self.model_name = model_name
        self.config = config
        self.precision = "fp32"
        self.batch_size = batch_size

        self.use_gpu = False
        self.use_xpu = False
        self.use_cpu = False

        if config.use_gpu:
            self.place = "gpu"
            self.use_gpu = True
        elif config.use_xpu:
            self.place = "xpu"
            self.use_xpu = True
        else:
            self.place = "cpu"
            self.use_cpu = True

        self.infer_time = 0
        self.samples = 0

        self.start = 0

        self.device_info = {"cpu_mem": None, "gpu_mem": None, "gpu_util": None}
        if mem_info is not None:
            self.mem_info = mem_info

    def tic(self):
        self.start = time.time()

    def toc(self, samples=1):
        self.infer_time += (time.time() - self.start) * 1000
        self.samples += samples

    def get_device_info(self, cpu_mem=None, gpu_mem=None, gpu_util=None):
        self.device_info["cpu_mem"] = cpu_mem
        self.device_info["gpu_mem"] = gpu_mem
        self.device_info["gpu_util"] = gpu_util

    def report(self):
        print("----------------------- Env info ------------------------")
        print("paddle_version: {}".format(paddle.__version__))
        print("----------------------- Model info ----------------------")
        print("model_name: {}".format(self.model_name))
        print("model_type: {}".format(self.precision))
        print("----------------------- Data info -----------------------")
        print("batch_size: {}".format(self.batch_size))
        print("num_of_samples: {}".format(self.samples))
        print("----------------------- Conf info -----------------------")
        print("runtime_device: {}".format(self.place))
        print("ir_optim: {}".format("True"
                                    if self.config.ir_optim() else "False"))
        print("enable_memory_optim: {}".format(
            "True" if self.config.enable_memory_optim() else "False"))
        if self.use_cpu:
            print("enable_mkldnn: {}".format(
                "True" if self.config.mkldnn_enabled() else "False"))
            print("cpu_math_library_num_threads: {}".format(
                self.config.cpu_math_library_num_threads()))
        print("----------------------- Perf info -----------------------")
        print(
            "[The average used CPU memory is (MB): {}. The average used GPU memory is (MB): {}. The average GPU util is : {}%".
            format(self.device_info['cpu_mem'], self.device_info['gpu_mem'],
                   self.device_info['gpu_util']))
        print("average_latency(ms): {}".format(self.infer_time / (self.samples
                                                                  )))
        print("QPS: {}".format((self.samples) / (self.infer_time / 1000.0)))

    @staticmethod
    def get_current_memory_mb(gpu_id=None):
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        cpu_mem = info.uss / 1024. / 1024.
        gpu_mem = 0
        if gpu_id is not None:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = meminfo.used / 1024. / 1024.
        return cpu_mem, gpu_mem

    @staticmethod
    def get_current_gputil(gpu_id):
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[gpu_id].load
        return gpu_load
