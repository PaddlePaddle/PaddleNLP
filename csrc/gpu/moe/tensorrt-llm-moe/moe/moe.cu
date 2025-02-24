
#pragma once

#include <optional>
#include <algorithm>
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "cutlass_helper.h"
#include "utils.h"
#include "profile.h"

// profile部分 ***************************************



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
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
        mProfiler->init(*mKernelRunner, mProfiler->mGemmToProfile,
            mActivationDtype,
            mWeightDtype,
            mOutputDtype, num_experts, top_k, hidden_size, inter_size,
            /* bias */ false, parallelism_config, mIsWeightOnlyIn4, QuantMethod);

        char* profile_workspace = nullptr;
        size_t tmp_workspace_size = mProfiler->getWorkspaceSize(mMaxDimM);
        auto const cu_malloc_status = cudaMalloc(&profile_workspace, tmp_workspace_size);
        
        
        if (cu_malloc_status != cudaSuccess) {
            std::cout << "Can't allocate tmp workspace for MOE GEMM tactics profiling." << std::endl;
        }

        for (auto const& m : num_token_buckets)
        {
            ProfileId best_profile_id = runProfileM(m, profile_workspace, stream);
            mMNKProfileMap->getMProfileMap(gemm_id_moe)->insert({m, best_profile_id});
        }

        auto const cu_free = cudaFree(profile_workspace);
        // TORCH_CHECK(cu_free == cudaSuccess, "Can't free tmp workspace for MOE GEMM profiling.");
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
        // // 筛选符合sm >= 90的所有配置
        // bool is_sm90 = sm >= 90;
        // printf("sm is {%d}\n", sm);
        // auto it = std::remove_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { 
        //     return c.is_sm90 != is_sm90; // 移除所有不符合sm >= 90的配置
        // });
        // tactics.erase(it, tactics.end()); // 保留符合sm >= 90的配置
        return tactics;
    }

    float runSingleProfile(int64_t const m, Profile const& profile, char* profile_workspace, cudaStream_t stream)
    {
        constexpr int warmup = 5;
        constexpr int runs = 15;

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

        std::sort(num_token_buckets.begin(), num_token_buckets.end());
        mMinDimM = num_token_buckets.front();
        mMaxDimM = num_token_buckets.back();

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        // common::check_cuda_error(cudaStreamCreate(&stream));

        profiler_backend::GemmToProfile gemm_idxes[]
            = {profiler_backend::GemmToProfile::GEMM_1, profiler_backend::GemmToProfile::GEMM_2};

        for (auto const& gemm_idx : gemm_idxes)
        {
            runProfileGemmIdx(hidden_size, inter_size, num_experts, static_cast<int>(top_k), static_cast<int>(tp_size),
                static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank), num_token_buckets,
                gemm_idx, stream);
        }
        cudaStreamDestroy(stream);
        // common::check_cuda_error(cudaStreamDestroy(stream));
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

            json gemm_entry;
            gemm_entry["gemm_idx"] = static_cast<int>(gemm_id_moe.gemm_idx);
            gemm_entry["hidden_size"] = gemm_id_moe.hidden_size;
            gemm_entry["inter_size"] = gemm_id_moe.inter_size;
            gemm_entry["num_experts"] = gemm_id_moe.num_experts;
            gemm_entry["top_k"] = gemm_id_moe.top_k;

            json profile_ids;
            for (const auto& entry : *profile_map) {
                json profile_entry;
                profile_entry["token_num"] = entry.first;  // Save token_num
                profile_entry["profile_id"] = entry.second; // Save corresponding profile_id
                auto config = mAllProfiles[entry.second];
                profile_entry["tile_config"] = config.tile_config;
                profile_entry["split_k_style"] = config.split_k_style;
                profile_entry["split_k_factor"] = config.split_k_factor;
                profile_entry["stages"] = config.stages;
                // Now push the profile_entry directly
                profile_ids.push_back(profile_entry);
            }

            gemm_entry["profile_ids"] = profile_ids;
            root.push_back(gemm_entry);
        }

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
                                                int64_t num_tokens, 
                                                const GemmIDMoe& gemm_id_moe1,
                                                const GemmIDMoe& gemm_id_moe2)
{
    std::vector<int64_t> profile_ids;

    // 使用哈希表存储 token_num -> profile_id 映射
    std::unordered_map<int64_t, int> gemm1_profile_map;
    std::unordered_map<int64_t, int> gemm2_profile_map;

    for (const auto& gemm_entry : root) {
        int gemm_idx = gemm_entry["gemm_idx"];
        
        // Check if gemm_id_moe1 or gemm_id_moe2 match
        if (gemm_idx == static_cast<int>(gemm_id_moe1.gemm_idx)) {
            const auto& profile_ids_json = gemm_entry["profile_ids"];
            for (const auto& entry : profile_ids_json) {
                int64_t token_num = entry.value("token_num", -1);  // Default to -1 if not found
                int profile_id = entry["profile_id"];
                gemm1_profile_map[token_num] = profile_id;
            }
        }
        else if (gemm_idx == static_cast<int>(gemm_id_moe2.gemm_idx)) {
            const auto& profile_ids_json = gemm_entry["profile_ids"];
            for (const auto& entry : profile_ids_json) {
                int64_t token_num = entry.value("token_num", -1);  // Default to -1 if not found
                int profile_id = entry["profile_id"];
                gemm2_profile_map[token_num] = profile_id;
            }
        }
    }

    // 查找 gemm1 和 gemm2 的 profile_id，确保它们的顺序
    if (gemm1_profile_map.find(num_tokens) != gemm1_profile_map.end()) {
        profile_ids.push_back(gemm1_profile_map[num_tokens]);
    } else {
        std::cerr << "No profile_id found for gemm1 with token_num " << num_tokens << std::endl;
    }

    if (gemm2_profile_map.find(num_tokens) != gemm2_profile_map.end()) {
        profile_ids.push_back(gemm2_profile_map[num_tokens]);
    } else {
        std::cerr << "No profile_id found for gemm2 with token_num " << num_tokens << std::endl;
    }
    std::cout <<"我又修改了stage 5" << std::endl;
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
tensorrt_llm::ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return tensorrt_llm::ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return tensorrt_llm::ActivationType::Geglu;
    }
    else if (activation_type_str == "Swiglu") {
        return tensorrt_llm::ActivationType::Swiglu;
    }
    else {
        std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
    }
    return tensorrt_llm::ActivationType::InvalidType;
}

kernels::MOEExpertScaleNormalizationMode getNormalizationMode(int normalization_mode) {
    switch (normalization_mode) {
        case 0:
            return kernels::MOEExpertScaleNormalizationMode::NONE;
        case 1:
            return kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;
        case 2:
            return kernels::MOEExpertScaleNormalizationMode::SPARSE_MIXER;
        case 3:
            return kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED;
        case 4:
            return kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED_RENORM;
        default:
            std::cerr << "Unknown normalization_mode value: " << normalization_mode << std::endl;
            // Return a default value if an invalid mode is passed
            return kernels::MOEExpertScaleNormalizationMode::NONE;
    }
}


template<typename T, typename WeightType>
Tensor trt_llm_fused_moe_helper(Tensor input_activations, 
                                 Tensor gating_output, 
                                 Tensor fc1_expert_weights, 
                                 tensorrt_llm::ActivationType fc1_activation_type,
                                 Tensor fc2_expert_weights, 
                                 const int active_rows, 
                                 const int k,
                                 paddle::optional<paddle::Tensor> scale1 = nullptr,
                                 paddle::optional<paddle::Tensor> scale2 = nullptr,
                                 paddle::optional<paddle::Tensor> scale3 = nullptr,
                                 int normalization_mode = 0,
                                 const std::string& quant_method = "none",
                                 int    tune_max_num_tokens=40960)
{
    typedef DataTypeMapper<T> traits_t;
    typedef typename traits_t::DataType DataType_;
    typedef typename traits_t::data_t data_t;

    typedef DataTypeMapper<WeightType> traits_w;
    typedef typename traits_w::DataType DataType_w;
    typedef typename traits_w::data_t data_w;

    const int num_rows = input_activations.shape()[0];//(num_tokens, hidden_size)
    const int hidden_size = input_activations.shape()[1];
    int inter_size = fc2_expert_weights.shape()[1]; //(num_experts, inter_size, hidden_size)

    if (quant_method == "fp8_block_wise") {
        std::cout << "我改了inter_size"<< std::endl;
       inter_size = fc2_expert_weights.shape()[2];
    }
    const int num_experts = gating_output.shape()[1]; //(num_tokens, num_experts)
    
    auto stream = input_activations.stream();
    auto place = input_activations.place();
    phi::Allocator* allocator = paddle::GetAllocator(place);
    T* input_act_ptr = reinterpret_cast<T*>(input_activations.data<data_t>());
    float* gating_output_ptr =  reinterpret_cast<float*>(gating_output.data<float>());

    // bias暂时先不支持
    T* fc1_expert_biases_ptr = nullptr;
    T* fc2_expert_biases_ptr = nullptr;

    bool* finished_ptr = nullptr;

    kernels::MOEParallelismConfig moe_parallel_config = kernels::MOEParallelismConfig(1, 0, 1, 0);

    void* scale1_ptr = nullptr;
    void* scale2_ptr = nullptr;

    bool use_deepseek = false;
    if (quant_method == "fp8_block_wise") {
        use_deepseek = true;
    }

    void* fc1_weights_ptr = nullptr;
    void* fc2_weights_ptr = nullptr;


    kernels::QuantParams quant_params;
    if (quant_method == "weight_only_int8" || quant_method == "weight_only_int4") {
        scale1_ptr = get_ptr<data_t>(scale1);
        scale2_ptr = get_ptr<data_t>(scale2);
        quant_params = kernels::QuantParams::Int(scale1_ptr, scale2_ptr);
        fc1_weights_ptr = reinterpret_cast<WeightType*>(fc1_expert_weights.data<data_w>());
        fc2_weights_ptr = reinterpret_cast<WeightType*>(fc2_expert_weights.data<data_w>());
    } else if (quant_method == "fp8_block_wise") {
        // fp8 scale是float
        scale1_ptr = get_ptr<float>(scale1);
        scale2_ptr = get_ptr<float>(scale2);;
        fc1_weights_ptr = reinterpret_cast<__nv_fp8_e4m3*>(fc1_expert_weights.data<phi::dtype::float8_e4m3fn>());
        fc2_weights_ptr = reinterpret_cast<__nv_fp8_e4m3*>(fc2_expert_weights.data<phi::dtype::float8_e4m3fn>());
         
        #ifdef MYDEBUG
        std::cout <<"打印一下weight 1000个" << std::endl;
        print_gpu_data<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(fc1_expert_weights.data<phi::dtype::float8_e4m3fn>()), num_experts * 2 * inter_size * hidden_size, 10000);
        #endif

    } else {
        fc1_weights_ptr = reinterpret_cast<WeightType*>(fc1_expert_weights.data<data_w>());
        fc2_weights_ptr = reinterpret_cast<WeightType*>(fc2_expert_weights.data<data_w>());
    }

    kernels::BlockScaleParams deepseek_params;
    if (use_deepseek)
    {    
        using BlockScaleGemmImplPtr = std::shared_ptr<kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;
        BlockScaleGemmImplPtr mBlockScaleGemmImplPtr;

        mBlockScaleGemmImplPtr = std::make_shared<kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
                    __nv_fp8_e4m3, __nv_bfloat16>>();;
        
        size_t deepseek_workspace_size = 0;
        cudaEvent_t mMemcpyEvent;
        bool is_gated_activation = isGatedActivation(fc1_activation_type);
        int factor = is_gated_activation ? 2 : 1;
        size_t deepseek_fc1_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, factor * inter_size, hidden_size, num_experts);
        size_t deepseek_fc2_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, hidden_size, inter_size, num_experts);
        deepseek_workspace_size = std::max(deepseek_fc1_size, deepseek_fc2_size);

        auto deepseek_ws = allocator->Allocate(deepseek_workspace_size)->ptr();
        
        // #ifdef MYDEBUG
        // print_gpu_data<float>( static_cast<float*>(scale1_ptr), num_experts * 2 * inter_size, num_experts * 2 * inter_size);
        // #endif

        deepseek_params = kernels::BlockScaleParams(
            static_cast<float *>(scale1_ptr), static_cast<float *>(scale2_ptr), mBlockScaleGemmImplPtr, reinterpret_cast<char*>(deepseek_ws), &mMemcpyEvent);
    }

    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner_ptr = std::make_shared<kernels::CutlassMoeFCRunner<T, WeightType>>();
    std::string profile_file = "./trt_moe_profile_results.json";
    CutlassGemmConfigMannager& best_config_mannager = CutlassGemmConfigMannager::getInstance();

    if (getenv("FLAGS_efficientllm_op_configs")) {
        std::string efficientllm_op_configs = getenv("FLAGS_efficientllm_op_configs");
        if (efficientllm_op_configs == "tune")  {
            std::cout <<"2 : " <<   tune_max_num_tokens << std::endl;
            FusedMoeRunnerPofiler profiler(/* activation_dtype= */ input_activations.dtype(), 
                                    /* weight_dtype= */ fc1_expert_weights.dtype(), 
                                    /* output_dtype= */ input_activations.dtype(), 
                                    /* moe_runner= */ moe_runner_ptr, 
                                    /* quant_method= */ quant_method);

            // std::vector<int64_t> num_token_buckets = get_power_of_2_num_tokens_buckets(tune_max_num_tokens);
            // std::cout <<"num_token_buckets : " <<   tune_max_num_tokens << std::endl;

            std::vector<int64_t> num_token_buckets = {1024};
            std::cout << "我只tune 1024"<< std::endl;
            profiler.runProfile(fc2_expert_weights, k, 1, 0, 1, 0, num_token_buckets);
            // 需要将profile的结果，即num_tokens和对应的profile_ids落在本地efficientllm_op_configs路径下，这里需要补充代码
            profiler.saveProfileResultsToFile(profile_file);

            std::vector<int64_t> profile_ids = profiler.getProfileIds(next_positive_power_of_2(num_rows), fc2_expert_weights, k, num_experts);

            std::cout << "profile_ids : " << profile_ids.size() << std::endl;

            for (int i = 0; i < profile_ids.size(); ++i){
                std::cout <<  "profile_ids : "<< profile_ids[i] << std::endl;
            }

            setRunnerProfiles(moe_runner_ptr, profile_ids, quant_method);
        } else {
            if (!std::filesystem::exists(efficientllm_op_configs)) {
                PADDLE_THROW(phi::errors::Fatal("Warning: The file \"" + efficientllm_op_configs + "\" does not exist in the specified path." ));
            } else {
                // 从config文件路径读取到适合当前num_tokens的配置，这里需要重写一个方法，可以根据num_rows读取到对应配置
                auto gemm_id_moe1 = GemmIDMoe{profiler_backend::GemmToProfile::GEMM_1, hidden_size, inter_size,
                    static_cast<int>(num_experts), static_cast<int>(k)};
                auto gemm_id_moe2 = GemmIDMoe{profiler_backend::GemmToProfile::GEMM_2, hidden_size, inter_size,
                    static_cast<int>(num_experts), static_cast<int>(k)};
                auto* configs_json = best_config_mannager.get_gemm_best_configs(profile_file);

                std::vector<int64_t> profile_ids = loadProfileResultsFromFile(*configs_json, next_positive_power_of_2(num_rows), gemm_id_moe1, gemm_id_moe2);
                setRunnerProfiles(moe_runner_ptr, profile_ids, quant_method);
            }
        }
    } else {
        printf("Warning :Moe not tune cutlass kernel, using defualt config!\n");
        auto [tactic1, tactic2] = selectTacticsForArch(moe_runner_ptr);
        moe_runner_ptr->setTactic(std::make_optional(tactic1), std::make_optional(tactic2));
    }
    
    
    // std::vector<int64_t> profile_ids = {20, 19};
    // setRunnerProfiles(moe_runner_ptr, profile_ids, quant_method);
    // std::cout <<"我设置了tatic 20 19" << std::endl;


    kernels::MOEExpertScaleNormalizationMode normalization_mode_enum = getNormalizationMode(normalization_mode);

    auto bytes = moe_runner_ptr->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
                                             normalization_mode_enum, moe_parallel_config, use_deepseek);

    auto workspace_ptr = allocator->Allocate(bytes)->ptr();
    auto expert_scales = paddle::empty({num_rows, k}, input_activations.dtype(), place);
    T* expert_scales_ptr = reinterpret_cast<T*>(expert_scales.data<data_t>());

    auto expanded_source_row_to_expanded_dest_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expanded_source_row_to_expanded_dest_row_ptr = reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.data<int>());

    // topk的结果
    auto expert_for_source_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expert_for_source_row_ptr = reinterpret_cast<int*>(expert_for_source_row.data<int>());

    auto output_tensor = paddle::empty({num_rows, hidden_size}, input_activations.dtype(), place);
    T* output_tensor_ptr = reinterpret_cast<T*>(output_tensor.data<data_t>());

    moe_runner_ptr->runMoe(input_act_ptr,
                    gating_output_ptr,
                    fc1_weights_ptr,
                    fc1_expert_biases_ptr, // nullptr
                    fc1_activation_type,
                    fc2_weights_ptr,
                    fc2_expert_biases_ptr, // nullptr
                    quant_params,
                    num_rows,
                    hidden_size,
                    inter_size,
                    num_experts,
                    k,
                    reinterpret_cast<char*>(workspace_ptr),
                    output_tensor_ptr,
                    finished_ptr,
                    active_rows,
                    expert_scales_ptr,
                    expanded_source_row_to_expanded_dest_row_ptr,
                    expert_for_source_row_ptr,
                    0.2f,  // sparse_mixer_epsilon
                    moe_parallel_config,
                    normalization_mode_enum,
                    use_deepseek,
                    deepseek_params,
                    stream);
    return output_tensor;
}

template<typename T, typename WeightType>
Tensor trt_llm_fused_moe_helper_fp8_per_tensor(Tensor input_activations, 
                                 Tensor gating_output, 
                                 Tensor fc1_expert_weights, 
                                 tensorrt_llm::ActivationType fc1_activation_type,
                                 Tensor fc2_expert_weights, 
                                 const int active_rows, 
                                 const int k,
                                 paddle::optional<paddle::Tensor> scale1 = nullptr,
                                 paddle::optional<paddle::Tensor> scale2 = nullptr,
                                 paddle::optional<paddle::Tensor> scale3 = nullptr,
                                 const std::string& quant_method = "none",
                                 int     tune_max_num_tokens=40960)
{
    typedef DataTypeMapper<T> traits_t;
    typedef typename traits_t::DataType DataType_;
    typedef typename traits_t::data_t data_t;

    typedef DataTypeMapper<WeightType> traits_w;
    typedef typename traits_w::DataType DataType_w;
    typedef typename traits_w::data_t data_w;

    const int num_rows = input_activations.shape()[0];
    const int hidden_size = input_activations.shape()[1];
    const int inter_size = fc2_expert_weights.shape()[1];
    const int num_experts = gating_output.shape()[1];
    auto stream = input_activations.stream();
    auto place = input_activations.place();

    data_t* input_act_ptr = get_ptr<data_t>(input_activations);
    float* gating_output_ptr = get_ptr<float>(gating_output);

    float* scale1_ptr = scale1 ? get_ptr<float>(scale1) : nullptr;
    float* scale2_ptr = scale2 ? get_ptr<float>(scale2) : nullptr;
    float* scale3_ptr = scale3 ? get_ptr<float>(scale3) : nullptr;

    data_w* fc1_expert_weights_ptr = get_ptr<data_w>(fc1_expert_weights);
    data_t* fc1_expert_biases_ptr = nullptr;

    data_w* fc2_expert_weights_ptr = get_ptr<data_w>(fc2_expert_weights);
    data_t* fc2_expert_biases_ptr = nullptr;

    bool* finished_ptr = nullptr;

    kernels::MOEParallelismConfig moe_parallel_config = kernels::MOEParallelismConfig(1, 0, 1, 0);

    // 根据启用的量化方法设置量化参数
    kernels::QuantParams quant_params;
    quant_params = kernels::QuantParams::FP8(scale1_ptr, scale2_ptr, scale3_ptr);

    int sm = getSMVersion();
    kernels::CutlassMoeFCRunner<T, WeightType, __nv_bfloat16> moe_runner;
    kernels::BlockScaleParams deepseek_params;
    bool use_deepseek = false;

    auto [tactic1, tactic2] = selectTacticsForArch(moe_runner, sm);
    moe_runner.setTactic(std::make_optional(tactic1), std::make_optional(tactic2));

    auto bytes = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
                                             kernels::MOEExpertScaleNormalizationMode::RENORMALIZE, moe_parallel_config, false);

    auto workspace_tensor = paddle::empty({static_cast<int>(bytes)}, paddle::DataType::UINT8, place);
    uint8_t* uint8_ptr = get_ptr<uint8_t>(workspace_tensor);
    char* workspace_ptr = reinterpret_cast<char*>(uint8_ptr);

    auto fc2_output = paddle::empty({k * num_rows, hidden_size}, input_activations.dtype(), place);
    auto expert_scales = paddle::empty({num_rows, k}, input_activations.dtype(), place);
    data_t* expert_scales_ptr = get_ptr<data_t>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

    auto output_tensor = paddle::empty({num_rows, hidden_size}, input_activations.dtype(), place);
    data_t* output_tensor_ptr = get_ptr<data_t>(output_tensor);

    moe_runner.runMoe(input_act_ptr,
                      gating_output_ptr,
                      fc1_expert_weights_ptr,
                      fc1_expert_biases_ptr,
                      fc1_activation_type,
                      fc2_expert_weights_ptr,
                      fc2_expert_biases_ptr,
                      quant_params,
                      num_rows,
                      hidden_size,
                      inter_size,
                      num_experts,
                      k,
                      workspace_ptr,
                      output_tensor_ptr,
                      finished_ptr,
                      active_rows,
                      expert_scales_ptr,
                      expanded_source_row_to_expanded_dest_row_ptr,
                      expert_for_source_row_ptr,
                      0.2f,  // sparse_mixer_epsilon
                      moe_parallel_config,
                      kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,
                      use_deepseek,
                      deepseek_params,
                      stream);

    return output_tensor;
}

std::vector<paddle::Tensor> TrtLLMFusedMoe(const paddle::Tensor&     input_activations, //(num_tokens, hidden_size)
                const paddle::Tensor&      gating_output, //(num_tokens, num_experts)
                const paddle::Tensor&      fc1_expert_weights, //(num_experts, hidden_size, inter_size * 2)
                const paddle::Tensor&      fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                const paddle::optional<paddle::Tensor>& scale1,
                const paddle::optional<paddle::Tensor>& scale2,
                const paddle::optional<paddle::Tensor>& scale3,
                int     k,
                int normalization_mode = 1,
                const std::string& quant_method="none",
                const std::string& fc1_activation_type_str="Swiglu",
                int     tune_max_num_tokens=40960)
{

    const auto _st = input_activations.dtype();
    const auto weight_type = fc1_expert_weights.dtype();
    const int num_rows    = input_activations.shape()[0];
    const auto quant_type = fc2_expert_weights.dtype();

    Tensor output_tensor;
    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::Swiglu;;
    std::cout << "start ! "<< std::endl;
    std::cout<< quant_method  << std::endl;
    switch (_st) {
        case paddle::DataType::FLOAT16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<half, half>(input_activations,
                                                                    gating_output,
                                                                    fc1_expert_weights,
                                                                    fc1_activation_type,
                                                                    fc2_expert_weights,
                                                                    num_rows,
                                                                    k,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    normalization_mode,
                                                                    quant_method);
            }
            else {
                std::string err_msg = "Unsupported weight type ";
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case paddle::DataType::BFLOAT16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                num_rows,
                                                                                k,
                                                                                nullptr,
                                                                                nullptr,
                                                                                nullptr,
                                                                                normalization_mode,
                                                                                quant_method,
                                                                                tune_max_num_tokens);
            } else {
                if (quant_method == "weight_only_int8") {
                    output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                num_rows,
                                                                                k,
                                                                                scale1,
                                                                                scale2,
                                                                                nullptr, //scale3不需要
                                                                                normalization_mode,
                                                                                quant_method,
                                                                                tune_max_num_tokens);
                } else if (quant_method == "weight_only_int4") {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                                            gating_output,
                                                                            fc1_expert_weights,
                                                                            fc1_activation_type,
                                                                            fc2_expert_weights,
                                                                            num_rows,
                                                                            k,
                                                                            scale1,
                                                                            scale2,
                                                                            nullptr, //scale3不需要
                                                                            normalization_mode,
                                                                            quant_method,
                                                                            tune_max_num_tokens);
                } else if (quant_method == "fp8_block_wise") {
                    output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_fp8_e4m3>(input_activations,
                                                                            gating_output,
                                                                            fc1_expert_weights,
                                                                            fc1_activation_type,
                                                                            fc2_expert_weights,
                                                                            num_rows,
                                                                            k,
                                                                            scale1,
                                                                            scale2,
                                                                            nullptr, //scale3不需要
                                                                            normalization_mode,
                                                                            quant_method,
                                                                            tune_max_num_tokens);
                } else {
                    std::string err_msg = "Unsupported weight type ";
                    throw std::runtime_error(err_msg);
                }
            }
            break;
        }
        case paddle::DataType::FLOAT8_E4M3FN: {
            std::cout << "fp8_per_tensor"<< std::endl;
            // if (quant_type == _st) {
            //     output_tensor = trt_llm_fused_moe_helper_fp8_per_tensor<__nv_fp8_e4m3, __nv_fp8_e4m3>(input_activations,
            //                                                                     gating_output,
            //                                                                     fc1_expert_weights,
            //                                                                     fc1_activation_type,
            //                                                                     fc2_expert_weights,
            //                                                                     num_rows,
            //                                                                     k,
            //                                                                     scale1,
            //                                                                     scale2,
            //                                                                     scale3,
            //                                                                     quant_method,
                                                                                    // tune_max_num_tokens);
            // }
            // else {
            //     std::string err_msg = "Unsupported weight type ";
            //     throw std::runtime_error(err_msg);
            // }
            break;
        }
        
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return {output_tensor};
}


PD_BUILD_OP(trt_llm_fused_moe)
    .Inputs({"input_activations", "gating_output", "fc1_expert_weights", "fc2_expert_weights", paddle::Optional("scale1"), paddle::Optional("scale2"), paddle::Optional("scale3"),})
    .Outputs({"output_tensor"})
    .Attrs({"k: int", "normalization_mode: int", "quant_method:std::string", "fc1_activation_type_str:std::string", "tune_max_num_tokens: int"})
    .SetKernelFn(PD_KERNEL(TrtLLMFusedMoe));