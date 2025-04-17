# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import heapq
import os
from typing import Callable

import paddle
import triton
from triton import Config, cdiv
from triton.runtime import autotuner, driver
from triton.testing import (
    get_dram_gbps,
    get_max_simd_tflops,
    get_max_tensorcore_tflops,
    nvsmi,
)

_AUTOTUNE: bool = os.getenv("CCE_AUTOTUNE", "0") != "0"


@functools.lru_cache()
def get_clock_rate_in_khz():
    try:
        return nvsmi(["clocks.max.sm"])[0] * 1e3
    except FileNotFoundError:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps) / num_subcores * get_max_simd_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_tflops(device, num_ctas, num_warps, dtype):
    capability = paddle.device.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == paddle.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(device, num_ctas, num_warps, dtype)


def early_config_prune(
    configs,
    named_args,
    *,
    shared_memory_factor: float = 1.0,
    max_num_warps: int = None,
    **kwargs,
):
    device = paddle.get_device()
    device = int(device.split(":")[-1])
    capability = paddle.device.cuda.get_device_capability()
    # BLOCK_B, BLOCK_V, BLOCK_D, SPLIT_K, num_warps, num_stages
    dtsize = named_args["E"].element_size()

    if max_num_warps is not None:
        configs = [config for config in configs if config.num_warps <= max_num_warps]

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_B, BLOCK_V, BLOCK_D, num_stages = (
            kw["BLOCK_B"],
            kw["BLOCK_V"],
            kw["BLOCK_D"],
            config.num_stages,
        )

        max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        required_shared_memory = shared_memory_factor * (BLOCK_B + BLOCK_V) * BLOCK_D * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        pruned_configs.append(config)

    configs = pruned_configs

    # group configs by (BLOCK_B,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_B, BLOCK_V, BLOCK_D, num_warps, num_stages = (
            kw["BLOCK_B"],
            kw["BLOCK_V"],
            kw["BLOCK_D"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_B, BLOCK_V, BLOCK_D, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_B, BLOCK_V, BLOCK_D, num_warps = k
        if capability[0] >= 8:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_B * BLOCK_V * BLOCK_D / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: 10 + abs(x[1] - optimal_num_stages)
                if (x[1] - optimal_num_stages) < 0
                else x[1] - optimal_num_stages,
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs


def _total_ops_fn(B, V, D) -> float:
    return 2 * B * V * D + 10 * B * V


def _total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v):
    return B * dtsize


def estimate_matmul_time(
    # backend, device,
    num_warps,
    num_stages,  #
    E,
    B,
    V,
    D,  #
    BLOCK_B,
    BLOCK_V,
    BLOCK_D,
    debug=False,
    total_ops_fn=_total_ops_fn,
    total_store_fn=_total_store_fn,
    **kwargs,  #
):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    device = paddle.get_device()
    device = int(device.split(":")[-1])
    dtype = E.dtype
    dtsize = E.element_size()

    num_cta_b = cdiv(B, BLOCK_B)
    num_cta_v = cdiv(V, BLOCK_V)
    num_ctas = num_cta_b * num_cta_v

    # If the input is smaller than the block size
    B, V = max(B, BLOCK_B), max(V, BLOCK_V)

    # time to compute
    total_ops = total_ops_fn(B, V, D)
    total_ops = total_ops / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # time to load data
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)  # 32 active ctas are enough to saturate
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)  # 32-108, remaining 5%
    dram_bw = get_dram_gbps(device) * (active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05)  # in GB/s
    l2_bw = dram_bw * 4  # rough estimation (should be 4.7 for A100?)
    # assume 80% of (following) loads are in L2 cache
    load_a_dram = B * D * dtsize * (1 + 0.2 * (num_cta_v - 1))
    load_a_l2 = B * D * dtsize * 0.8 * (num_cta_v - 1)
    load_b_dram = V * D * dtsize * (1 + 0.2 * (num_cta_b - 1))
    load_b_l2 = V * D * dtsize * 0.8 * (num_cta_b - 1)
    # total
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)  # MB
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    # loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # estimate storing time
    store_bw = dram_bw * 0.4  # :o
    store_dram = total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v) / (1024 * 1024)
    store_ms = store_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(
            f"{BLOCK_B=}, {BLOCK_V=}, {BLOCK_D=}, {num_warps=}, {num_stages=}, "
            f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
            f"loading time: {load_ms}ms, store time: {store_ms}ms, "
            f"Activate CTAs: {active_cta_ratio*100}%"
        )
    return total_time_ms


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_B": block_m,
                                "BLOCK_V": block_n,
                                "BLOCK_D": block_k,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


def get_autotune_config():
    return [
        # basic configs for compute-bound matmuls
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128},
            num_stages=2,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 64, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 256, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 32, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config({"BLOCK_B": 64, "BLOCK_V": 32, "BLOCK_D": 32}, num_stages=5, num_warps=2),
        # good for int8
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 128},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 128},
            num_stages=3,
            num_warps=16,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 128},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 128},
            num_stages=3,
            num_warps=16,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 64, "BLOCK_D": 128},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 256, "BLOCK_D": 128},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 64},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 64},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 32, "BLOCK_D": 64},
            num_stages=4,
            num_warps=4,
        ),
        Config({"BLOCK_B": 64, "BLOCK_V": 32, "BLOCK_D": 64}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound()


def _heuristics_from_config(config: Config) -> Callable[..., autotuner.Heuristics]:
    return triton.heuristics({k: (lambda args, _v=v: _v) for k, v in config.all_kwargs().items()})


def _cce_forward_best_config() -> Config:
    return Config(dict(BLOCK_B=256, BLOCK_V=128, BLOCK_D=32), num_warps=8, num_stages=3)


def cce_forward_autotune() -> Callable[..., autotuner.PaddleAutotuner]:
    if _AUTOTUNE:
        return triton.paddle_autotune(
            configs=get_autotune_config(),
            key=["V", "D", "B_BIN"],
            prune_configs_by={
                "early_config_prune": early_config_prune,
                "perf_model": estimate_matmul_time,
                "top_k": 10,
            },
            restore_value=["LSE"],
        )
    else:
        return _heuristics_from_config(_cce_forward_best_config())


def _bw_total_ops_fn(B, V, D) -> float:
    return 2 * B * V * D + 6 * B * V + 0.2 * (2 * B * V * D + 2 * B * V * D)


def _bw_total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v):
    return 0.2 * (num_cta_v * B * D * dtsize + num_cta_b * D * V * dtsize)


def _cce_backward_best_config() -> Config:
    return Config(dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=4)


def cce_backward_autotune() -> Callable[..., autotuner.PaddleAutotuner]:
    if _AUTOTUNE:
        return triton.paddle_autotune(
            configs=get_autotune_config(),
            key=["V", "D", "B_BIN"],
            prune_configs_by={
                "early_config_prune": functools.partial(early_config_prune, shared_memory_factor=2.0),
                "perf_model": functools.partial(
                    estimate_matmul_time,
                    total_ops_fn=_bw_total_ops_fn,
                    total_store_fn=_bw_total_store_fn,
                ),
                "top_k": 5,
            },
            reset_to_zero=["dE", "dC"],
        )
    else:
        return _heuristics_from_config(_cce_backward_best_config())


def _indexed_dot_best_config() -> Config:
    return Config(dict(BLOCK_B=128, BLOCK_D=256), num_warps=16, num_stages=4)


def _indexed_dot_all_configs() -> list[Config]:
    return [
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=128,
            ),
            num_warps=4,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=128,
            ),
            num_warps=8,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=256,
                BLOCK_D=256,
            ),
            num_warps=16,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=256,
                BLOCK_D=128,
            ),
            num_warps=16,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=256,
            ),
            num_warps=16,
            num_stages=4,
        ),
    ]


def indexed_dot_autotune() -> Callable[..., autotuner.PaddleAutotuner]:
    if _AUTOTUNE:
        return triton.paddle_autotune(
            configs=_indexed_dot_all_configs(),
            key=["D", "B_BIN"],
            reset_to_zero=["Out"],
        )
    else:
        return _heuristics_from_config(_indexed_dot_best_config())
