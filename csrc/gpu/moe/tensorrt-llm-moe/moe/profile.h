

#pragma once
#include "utils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"

using profiler_backend = kernels::GemmProfilerBackend;
using Profile = cutlass_extensions::CutlassGemmConfig;


int getSMVersion() {
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.major * 10 + props.minor;
}

std::vector<cutlass_extensions::CutlassGemmConfig> getFilteredConfigs(
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner, int sm) {
    std::vector<Profile> tactics = moe_runner->getTactics();
    std::cout <<"f性能差啊" << std::endl;
    if (sm == 89) {
        // Filter some unsupported configs for L40S
        auto it = std::remove_if(tactics.begin(), tactics.end(),
            [&](auto conf) {
                using cutlass_extensions::CutlassTileConfig;
                auto checks = std::vector{
                    // Fail for BF16/FP16
                    conf.tile_config == CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
                    conf.tile_config == CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64 && conf.stages == 4,
                    // Fail for FP8
                    false && conf.tile_config == CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128
                        && conf.stages >= 3,
                };

                return std::any_of(checks.begin(), checks.end(), [](auto v) { return v; });
            });
        tactics.erase(it, tactics.end());
    }

    if (tactics.empty()) {
        throw std::runtime_error("No valid GEMM tactics found");
    }

    return tactics;
}



std::pair<cutlass_extensions::CutlassGemmConfig, cutlass_extensions::CutlassGemmConfig> 
selectTacticsForArch(std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner) {
    int sm = getSMVersion();
    bool is_sm90 = sm >= 90;
    auto tactics = getFilteredConfigs(moe_runner, sm);
    auto it = std::find_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { return c.is_sm90 == is_sm90; });
    if (it == tactics.end()) {
        // Fall back to any tactic
        std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
        return std::make_pair(tactics[0], tactics[0]);
    }

    return std::make_pair(*it, *it);
}
