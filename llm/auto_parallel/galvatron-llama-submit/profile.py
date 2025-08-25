from paddlenlp.experimental.galvatron.profiler.model_profiler import ModelProfiler, ModelProfilerArguments
from paddlenlp.experimental.galvatron.utils import get_current_all_args

if __name__ == '__main__':
    args_dict = get_current_all_args()
    model_profiler_args = ModelProfilerArguments()
    model_profiler_args.initialize(args_dict=args_dict)
    model_profiler = ModelProfiler(model_profiler_args, args_dict)
    model_profiler.launch_profiling()
    model_profiler.process_data()