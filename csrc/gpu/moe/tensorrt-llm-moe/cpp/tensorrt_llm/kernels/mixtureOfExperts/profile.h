

#pragma once
#include "utils.h"
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


// ***************************************profile部分 ***************************************
struct GemmIDMoe
{
    profiler_backend::GemmToProfile gemm_idx;
    int64_t hidden_size;
    int64_t inter_size;
    int num_experts;
    int top_k;

    bool operator==(GemmIDMoe const& id) const
    {
        return id.gemm_idx == gemm_idx && id.hidden_size == hidden_size && id.inter_size == inter_size
            && id.num_experts == num_experts && id.top_k == top_k;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIDMoe const& id)
    {
        out << "gemm_idx, hidden_size, inter_size, num_experts, top_k=" << static_cast<int>(id.gemm_idx) << ","
            << id.hidden_size << "," << id.inter_size << "," << id.num_experts << "," << id.top_k;
        return out;
    }
};

struct GemmIDMoeHash
{
    std::size_t operator()(GemmIDMoe const& id) const
    {
        size_t hash = std::hash<int>{}(static_cast<int>(id.gemm_idx));
        hash ^= std::hash<int64_t>{}(id.hidden_size);
        hash ^= std::hash<int64_t>{}(id.inter_size);
        hash ^= std::hash<int>{}(id.num_experts);
        hash ^= std::hash<int>{}(id.top_k);
        return hash;
    }
};

using ProfileId = int;
using MProfileMap = std::unordered_map<int, ProfileId>;
using MProfileMapPtr = std::shared_ptr<MProfileMap>;

struct MNKProfileMap
{
    std::unordered_map<GemmIDMoe, MProfileMapPtr, GemmIDMoeHash> profile_map;

    bool existsMProfileMap(GemmIDMoe const& id)
    {
        auto const iter = profile_map.find(id);
        return iter != profile_map.end();
    }

    void createMProfileMap(GemmIDMoe const& id)
    {
        profile_map[id] = std::make_shared<MProfileMap>();
    }

    MProfileMapPtr getMProfileMap(GemmIDMoe const& id)
    {
        auto const iter = profile_map.find(id);
        if (iter == profile_map.end())
        {
            PADDLE_THROW("Cannot find ID  in the profile map. Abort.");
        }
        return iter->second;
    }
};


class FusedMoeRunnerPofiler {

public:
    FusedMoeRunnerPofiler(paddle::DataType activation_dtype, paddle::DataType weight_dtype, paddle::DataType output_dtype,
        std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner, std::string quant_method) {
            mActivationDtype = activation_dtype;
            mWeightDtype = weight_dtype;
            mOutputDtype = output_dtype;
            if (quant_method == "fp8_block_wise") {
                mUseFp8BlockScaling = true;
            } else if (quant_method == "weight_only_in4") {
                mIsWeightOnlyIn4 = true;
            }
            mKernelRunner = moe_runner;
            mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
            mMNKProfileMap = std::make_shared<MNKProfileMap>();
            mAllProfiles = getFilteredConfigs(mKernelRunner->getTactics(), getSMVersion());
            mMinDimM = -1;
            mMaxDimM = -1;
            QuantMethod = quant_method;
        }

    void runProfileGemmIdx(int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const top_k,
        int const tp_size, int const tp_rank, int const ep_size, int const ep_rank,
        std::vector<int64_t> const& num_token_buckets, profiler_backend::GemmToProfile const gemm_idx,
        cudaStream_t stream)
    {
        auto gemm_id_moe = GemmIDMoe{gemm_idx, hidden_size, inter_size, num_experts, top_k};

        if (mMNKProfileMap->existsMProfileMap(gemm_id_moe))
        {
            return;
        }

        mMNKProfileMap->createMProfileMap(gemm_id_moe);

        mProfiler->mGemmToProfile = gemm_idx;
        // TODO: support more dtypes and expert parallelism
        auto parallelism_config = kernels::MOEParallelismConfig(1, 0, 1, 0);
        mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
            mActivationDtype,
            mWeightDtype,
            mOutputDtype, num_experts, top_k, hidden_size, inter_size,
            /* bias */ false, parallelism_config, mIsWeightOnlyIn4, QuantMethod);

        char* profile_workspace = nullptr;
        size_t tmp_workspace_size = mProfiler->getWorkspaceSize(mMaxDimM);
        auto const cu_malloc_status = cudaMalloc(&profile_workspace, tmp_workspace_size);
        
        PADDLE_ENFORCE(cu_malloc_status == cudaSuccess, "Can't allocate tmp workspace for MOE GEMM tactics profiling.");
        
        if (cu_malloc_status != cudaSuccess) {
            std::cout << "Can't allocate tmp workspace for MOE GEMM tactics profiling." << std::endl;
        }

        for (auto const& m : num_token_buckets)
        {
            ProfileId best_profile_id = runProfileM(m, profile_workspace, stream);
            mMNKProfileMap->getMProfileMap(gemm_id_moe)->insert({m, best_profile_id});
        }

        auto const cu_free = cudaFree(profile_workspace);
        PADDLE_ENFORCE(cu_free == cudaSuccess, "Can't free tmp workspace for MOE GEMM profiling.");
    }

    std::vector<Profile> getFilteredConfigs(std::vector<Profile> tactics, int sm) {
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

    float runSingleProfile(int64_t const m, Profile const& profile, char* profile_workspace, cudaStream_t stream)
    {
        constexpr int warmup = 5;
        constexpr int runs = 20;
        // warmup
        for (int i = 0; i < warmup; ++i)
        {
            mProfiler->runProfiler(m, profile, profile_workspace, stream);
        }

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaStreamSynchronize(stream);
        cudaEventRecord(start, stream);

        // profile
        for (int i = 0; i < runs; ++i)
        {
            mProfiler->runProfiler(m, profile, profile_workspace, stream);
        }

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed;

        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return elapsed / runs;
    }

    ProfileId runProfileM(int64_t const m, char* profile_workspace, cudaStream_t stream)
    {
        mProfiler->prepare(m, profile_workspace, stream);
        float best_time = std::numeric_limits<float>::max();
        ProfileId best_profile_id;
        // std::cout <<< "**********" << m << "************************"<<std::endl; 
        for (int i = 0; i < static_cast<int>(mAllProfiles.size()); ++i)
        {
            auto const& profile = mAllProfiles[i];
            float candidate_time = std::numeric_limits<float>::max();
            try
            {
                candidate_time = runSingleProfile(m, profile, profile_workspace, stream);
                std::cout <<"i : " << i << std::endl;
                std::cout <<"candidate_time : " << candidate_time << std::endl;
                std::cout <<"tile_config : " << static_cast<int>(profile.tile_config) << std::endl;
                std::cout <<"stages : " << static_cast<int>(profile.stages) << std::endl;
            }
            catch (std::exception const& e)
            {
                std::ostringstream msg;
                msg << "Cannot profile configuration " << i << ": " << profile.toString() << "\n (for"
                    << " m=" << m << ")"
                    << ", reason: \"" << e.what() << "\". Skipped";
                cudaGetLastError(); // Reset the last cudaError to cudaSuccess.

                std::cout << "Error: " << msg.str() << std::endl;
                continue;
            }

            if (candidate_time < best_time)
            {
                best_time = candidate_time;
                best_profile_id = i;
            }
        }
        std::cout << "best_profile_id : " << best_profile_id << std::endl;
        std::cout << "best_time : " << best_time << std::endl;
        return best_profile_id;
    }

    void runProfile(Tensor const& fc2_expert_weights, int64_t const top_k, int64_t const tp_size,
        int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, std::vector<int64_t> num_token_buckets)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        if (mUseFp8BlockScaling)
        {
            return; // TODO
        }

        std::cout << "注意这里 " << std::endl;
        int64_t hidden_size = fc2_expert_weights.shape()[2];
        int64_t inter_size = fc2_expert_weights.shape()[1];

        int num_experts = static_cast<int>(fc2_expert_weights.shape()[0] * ep_size);
        std::cout << "num_experts : " << num_experts << std::endl;

        std::sort(num_token_buckets.begin(), num_token_buckets.end());
        mMinDimM = num_token_buckets.front();
        mMaxDimM = num_token_buckets.back();

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        profiler_backend::GemmToProfile gemm_idxes[]
            = {profiler_backend::GemmToProfile::GEMM_1, profiler_backend::GemmToProfile::GEMM_2};

        for (auto const& gemm_idx : gemm_idxes)
        {   
            std::cout << "********************* start gemm1&2 profile*****************"<< std::endl;
            runProfileGemmIdx(hidden_size, inter_size, num_experts, static_cast<int>(top_k), static_cast<int>(tp_size),
                static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank), num_token_buckets,
                gemm_idx, stream);
        }
        cudaStreamDestroy(stream);
    }


    std::vector<int64_t> getProfileIds(int64_t const num_tokens, Tensor const& fc2_expert_weights,
        int64_t const top_k, int64_t const num_experts)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        std::cout <<"start getProfileIds " << std::endl;
        int64_t hidden_size = fc2_expert_weights.shape()[2];
        int64_t inter_size = fc2_expert_weights.shape()[1];
        auto gemm_id_moe1 = GemmIDMoe{profiler_backend::GemmToProfile::GEMM_1, hidden_size, inter_size,
            static_cast<int>(num_experts), static_cast<int>(top_k)};
        
        auto gemm_id_moe2 = GemmIDMoe{profiler_backend::GemmToProfile::GEMM_2, hidden_size, inter_size,
            static_cast<int>(num_experts), static_cast<int>(top_k)};

        if (!mMNKProfileMap->existsMProfileMap(gemm_id_moe1) || !mMNKProfileMap->existsMProfileMap(gemm_id_moe2))
        {   
            printf("Not find configs from tuned configs...\n");
            return {};
        }

        int64_t capped_num_tokens = num_tokens;
        if (num_tokens < mMinDimM)
        {
            capped_num_tokens = mMinDimM;
        }
        else if (num_tokens > mMaxDimM)
        {
            capped_num_tokens = mMaxDimM;
        }

        int gemm1_profile_id = mMNKProfileMap->getMProfileMap(gemm_id_moe1)->at(capped_num_tokens);
        int gemm2_profile_id = mMNKProfileMap->getMProfileMap(gemm_id_moe2)->at(capped_num_tokens);
        std::vector<int64_t> profile_ids = {gemm1_profile_id, gemm2_profile_id};
        return profile_ids;
    }

    void saveProfileResultsToFile(const std::string& file_path) {
        json root;
        for (const auto& gemm_id_moe_entry : mMNKProfileMap->profile_map) {
            const GemmIDMoe& gemm_id_moe = gemm_id_moe_entry.first;
            const MProfileMapPtr& profile_map = gemm_id_moe_entry.second;

            // Create an entry for the current gemm_id_moe
            json gemm_entry;
            gemm_entry["gemm_idx"] = static_cast<int>(gemm_id_moe.gemm_idx);
            gemm_entry["hidden_size"] = gemm_id_moe.hidden_size;
            gemm_entry["inter_size"] = gemm_id_moe.inter_size;
            gemm_entry["num_experts"] = gemm_id_moe.num_experts;
            gemm_entry["top_k"] = gemm_id_moe.top_k;

            // Create a map to store profiles for this gemm_idx
            json m_profile;

            for (const auto& entry : *profile_map) {
                json profile_entry;
                profile_entry["profile_id"] = entry.second; // profile_id corresponding to token_num
                auto config = mAllProfiles[entry.second];
                profile_entry["tile_config"] = config.tile_config;
                profile_entry["split_k_style"] = config.split_k_style;
                profile_entry["split_k_factor"] = config.split_k_factor;
                profile_entry["stages"] = config.stages;

                // Add the profile entry under the token number in the m_profile map
                m_profile[std::to_string(entry.first)] = profile_entry;
            }

            // Add the m_profile to the gemm_entry
            gemm_entry["m_profile"] = m_profile;

            // Add the gemm_entry to the root JSON array
            root.push_back(gemm_entry);
        }

        // Write the JSON object to the specified file
        std::ofstream file(file_path);
        if (file.is_open()) {
            file << root;
            file.close();
            std::cout << "Profile results saved to " << file_path << std::endl;
        } else {
            std::cerr << "Failed to save profile results to " << file_path << std::endl;
        }
}

private:
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
    std::shared_ptr<kernels::GemmProfilerBackend> mProfiler;
    std::shared_ptr<MNKProfileMap> mMNKProfileMap;
    int64_t mMinDimM;
    int64_t mMaxDimM;
    paddle::DataType mActivationDtype;
    paddle::DataType mWeightDtype;
    paddle::DataType mOutputDtype;
    bool mUseFp8BlockScaling = false;
    bool mIsWeightOnlyIn4 = false;
    std::string QuantMethod;

    std::mutex mMutex;

    using Profile = cutlass_extensions::CutlassGemmConfig;
    std::vector<Profile> mAllProfiles;

};

std::vector<int64_t> loadProfileResultsFromFile(const nlohmann::json& root, 
                                                int64_t num_tokens)
{
    std::vector<int64_t> profile_ids{0,0};

    // Iterate over the root JSON to extract the necessary information
    for (const auto& gemm_entry : root) {
        int gemm_idx = gemm_entry["gemm_idx"];

        // Check if this gemm_idx matches gemm_id_moe1 or gemm_id_moe2
        if (gemm_idx == static_cast<int>(profiler_backend::GemmToProfile::GEMM_1)) {
            // If it matches gemm_id_moe1, try to find the profile using token_num
            const auto& m_profile = gemm_entry["m_profile"];
            if (m_profile.contains(std::to_string(num_tokens))) {
                int profile_id = m_profile[std::to_string(num_tokens)]["profile_id"];
                profile_ids[0] = profile_id;
            } else {
                std::cerr << "No profile_id found for gemm1 with token_num " << num_tokens << std::endl;
            }
        }
        else if (gemm_idx == static_cast<int>(profiler_backend::GemmToProfile::GEMM_2)) {
            // If it matches gemm_id_moe2, try to find the profile using token_num
            const auto& m_profile = gemm_entry["m_profile"];
            if (m_profile.contains(std::to_string(num_tokens))) {
                int profile_id = m_profile[std::to_string(num_tokens)]["profile_id"];
                profile_ids[1] = profile_id;
            } else {
                std::cerr << "No profile_id found for gemm2 with token_num " << num_tokens << std::endl;
            }
        }
    }

    return profile_ids;
}


void setRunnerProfiles(std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner, std::vector<int64_t> profile_ids, const std::string& quant_method)
    {
        if (quant_method == "fp8_block_wise")
        {
            auto config = cutlass_extensions::CutlassGemmConfig(
                cutlass_extensions::CutlassTileConfigSM90::CtaShape128x16x128B,
                cutlass_extensions::MainloopScheduleType::AUTO,
                cutlass_extensions::EpilogueScheduleType::AUTO,
                cutlass_extensions::ClusterShape::ClusterShape_1x1x1);
            moe_runner->setTactic(config, config);
            return;
        }

        std::vector<Profile> mAllProfiles = moe_runner->getTactics();
        auto best_gemm1_profile = mAllProfiles.front();
        auto best_gemm2_profile = mAllProfiles.front();
        if (!profile_ids.empty())
        {   
            best_gemm1_profile = mAllProfiles.at(profile_ids[0]);
            best_gemm2_profile = mAllProfiles.at(profile_ids[1]);
        }
        moe_runner->setTactic(best_gemm1_profile, best_gemm2_profile);
    }

// profile部分 ***************************************
