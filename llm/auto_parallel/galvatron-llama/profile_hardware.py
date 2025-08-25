from paddlenlp.experimental.galvatron.profiler.hardware_profiler import HardwareProfiler, HardwareProfilerArgs
from paddlenlp.experimental.galvatron.utils import get_current_all_args
import os

if __name__ == '__main__':
    args_dict = get_current_all_args()
    print(f'[linguangming] all_agrs is {args_dict}')
    hardware_profiler_args = HardwareProfilerArgs()
    hardware_profiler_args.initialize(args_dict)
    profiler = HardwareProfiler(hardware_profiler_args)
    execution_path = os.getcwd()
    profiler.set_execution_path(execution_path)

    profiler.profile_allreduce()
    profiler.profile_p2p()
    profiler.profile_overlap()
    profiler.profile_sp()
    # profiler.remove_files()