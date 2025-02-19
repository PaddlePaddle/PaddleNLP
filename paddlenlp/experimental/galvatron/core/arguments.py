import argparse
from .search_engine.arguments import galvatron_search_args
from .profiler.arguments import galvatron_profile_hardware_args

def initialize_galvatron(model_args = None, mode="train_dist"):
    if mode == "search":
        extra_args_provider = [galvatron_search_args]
    elif mode == "profile_hardware":
        extra_args_provider = [galvatron_profile_hardware_args]
    if model_args is not None:
        extra_args_provider.append(model_args)
    args = parse_args(extra_args_provider)
    return args

def parse_args(extra_args_provider):
    parser = argparse.ArgumentParser()
    # Custom arguments.
    if extra_args_provider is not None:
        if isinstance(extra_args_provider, list):
            for extra_args in extra_args_provider:
                parser = extra_args(parser)
        else:
            parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args