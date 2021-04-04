#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>
#include "fastertransformer/cuda/cub/cub.cuh"


#include "fusion_decoding_op.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/enforce.h"
#include "pd_traits.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class FusionDecodingKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto stream = ctx.cuda_device_context().stream();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::CUDADeviceContext>();

    auto* input = ctx.Input<Tensor>("Input");

    auto* memory_sequence_length = ctx.Input<Tensor>("MemSeqLen");
    auto* word_emb = ctx.Input<Tensor>("WordEmbedding");

    auto self_layernorm_weight = ctx.MultiInput<Tensor>("SelfLayernormWeight");
    auto self_layernorm_bias = ctx.MultiInput<Tensor>("SelfLayernormBias");
    auto self_attn_query_weight = ctx.MultiInput<Tensor>("SelfQueryWeight");
    auto self_attn_query_bias = ctx.MultiInput<Tensor>("SelfQueryBias");
    auto self_attn_key_weight = ctx.MultiInput<Tensor>("SelfKeyWeight");
    auto self_attn_key_bias = ctx.MultiInput<Tensor>("SelfKeyBias");
    auto self_attn_value_weight = ctx.MultiInput<Tensor>("SelfValueWeight");
    auto self_attn_value_bias = ctx.MultiInput<Tensor>("SelfValueBias");
    auto self_attn_output_weight = ctx.MultiInput<Tensor>("SelfOutWeight");
    auto self_attn_output_bias = ctx.MultiInput<Tensor>("SelfOutBias");

    auto cross_layernorm_weight =
        ctx.MultiInput<Tensor>("CrossLayernormWeight");
    auto cross_layernorm_bias = ctx.MultiInput<Tensor>("CrossLayernormBias");
    auto cross_attn_query_weight = ctx.MultiInput<Tensor>("CrossQueryWeight");
    auto cross_attn_query_bias = ctx.MultiInput<Tensor>("CrossQueryBias");
    auto cross_attn_key_weight = ctx.MultiInput<Tensor>("CrossKeyWeight");
    auto cross_attn_key_bias = ctx.MultiInput<Tensor>("CrossKeyBias");
    auto cross_attn_value_weight = ctx.MultiInput<Tensor>("CrossValueWeight");
    auto cross_attn_value_bias = ctx.MultiInput<Tensor>("CrossValueBias");
    auto cross_attn_output_weight = ctx.MultiInput<Tensor>("CrossOutWeight");
    auto cross_attn_output_bias = ctx.MultiInput<Tensor>("CrossOutBias");

    auto ffn_layernorm_weight = ctx.MultiInput<Tensor>("FFNLayernormWeight");
    auto ffn_layernorm_bias = ctx.MultiInput<Tensor>("FFNLayernormBias");
    auto ffn_intermediate_weight = ctx.MultiInput<Tensor>("FFNInterWeight");
    auto ffn_intermediate_bias = ctx.MultiInput<Tensor>("FFNInterBias");
    auto ffn_output_weight = ctx.MultiInput<Tensor>("FFNOutWeight");
    auto ffn_output_bias = ctx.MultiInput<Tensor>("FFNOutBias");

    auto* decoder_layernorm_weight =
        ctx.Input<Tensor>("DecoderLayernormWeight");
    auto* decoder_layernorm_bias = ctx.Input<Tensor>("DecoderLayernormBias");
    auto* embedding_weight = ctx.Input<Tensor>("EmbWeight");
    auto* embedding_bias = ctx.Input<Tensor>("EmbBias");
    auto* position_encoding_table = ctx.Input<Tensor>("PositionEncEmb");

    auto* output_ids = ctx.Output<Tensor>("OutputIds");
    auto* parent_ids = ctx.Output<Tensor>("ParentIds");
    auto* sequence_length = ctx.Output<Tensor>("SequenceLength");

    // Not used for now.
    std::string decoding_strategy = ctx.Attr<std::string>("decoding_strategy");
    int beam_width_ =
        (decoding_strategy == "beam_search") ? ctx.Attr<int>("beam_size") : 1;
    int candidate_num_ = (decoding_strategy == "topk_sampling" ||
                          decoding_strategy == "topp_sampling")
                             ? ctx.Attr<int>("topk")
                             : 1;
    float probability_threshold_ = (decoding_strategy == "topk_sampling" ||
                                    decoding_strategy == "topp_sampling")
                                       ? ctx.Attr<float>("topp")
                                       : 0.0;
    int64_t max_seq_len_ = ctx.Attr<int64_t>("max_len");
    int head_num_ = ctx.Attr<int>("n_head");
    int size_per_head_ = ctx.Attr<int>("size_per_head");
    int num_layer_ = ctx.Attr<int>("num_layer");
    int start_id_ = ctx.Attr<int>("bos_id");
    int end_id_ = ctx.Attr<int>("eos_id");
    float beam_search_diversity_rate_ =
        ctx.Attr<float>("beam_search_diversity_rate");

    typedef PDTraits<T> traits_;
    typedef typename traits_::DataType DataType_;

    auto input_dims = input->dims();
    int batch_size_ = (decoding_strategy == "beam_search")
                          ? input_dims[0] / beam_width_
                          : input_dims[0];
    const int memory_max_seq_len = input_dims[1];
    const int memory_hidden_dim = input_dims[2];
    const int vocab_size = word_emb->dims()[0];

    DecodingInitParam<DataType_> decoding_params;
    decoding_params.cublas_handle = dev_ctx.cublas_handle();

    decoding_params.output_ids = output_ids->mutable_data<int>(ctx.GetPlace());
    decoding_params.parent_ids = parent_ids->mutable_data<int>(ctx.GetPlace());
    decoding_params.sequence_length =
        sequence_length->mutable_data<int>(ctx.GetPlace());

    typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
    decoding_params.stream = stream;
    int device_id;
    cudaGetDevice(&device_id);
    fastertransformer::Allocator<AllocatorType::CUDA> allocator_(device_id);

    decoding_params.memory_tensor =
        reinterpret_cast<const DataType_*>(input->data<T>());
    decoding_params.memory_sequence_length =
        memory_sequence_length->data<int>();

    DecoderInitParam<DataType_>* params =
        new DecoderInitParam<DataType_>[num_layer_];

    for (int i = 0; i < num_layer_; i++) {
      params[i].stream = stream;
      params[i].cublas_handle = dev_ctx.cublas_handle();

      // self attn
      params[i].self_layernorm.gamma = reinterpret_cast<const DataType_*>(
          self_layernorm_weight[i]->data<T>());
      params[i].self_layernorm.beta =
          reinterpret_cast<const DataType_*>(self_layernorm_bias[i]->data<T>());
      // query
      params[i].self_attention.query_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_query_weight[i]->data<T>());
      params[i].self_attention.query_weight.bias =
          reinterpret_cast<const DataType_*>(
              self_attn_query_bias[i]->data<T>());
      // key
      params[i].self_attention.key_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_key_weight[i]->data<T>());
      params[i].self_attention.key_weight.bias =
          reinterpret_cast<const DataType_*>(self_attn_key_bias[i]->data<T>());
      // value
      params[i].self_attention.value_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_value_weight[i]->data<T>());
      params[i].self_attention.value_weight.bias =
          reinterpret_cast<const DataType_*>(
              self_attn_value_bias[i]->data<T>());
      // out proj
      params[i].self_attention.attention_output_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_output_weight[i]->data<T>());
      params[i].self_attention.attention_output_weight.bias =
          reinterpret_cast<const DataType_*>(
              self_attn_output_bias[i]->data<T>());

      // cross
      params[i].cross_layernorm.gamma = reinterpret_cast<const DataType_*>(
          cross_layernorm_weight[i]->data<T>());
      params[i].cross_layernorm.beta = reinterpret_cast<const DataType_*>(
          cross_layernorm_bias[i]->data<T>());
      // query
      params[i].cross_attention.query_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_query_weight[i]->data<T>());
      params[i].cross_attention.query_weight.bias =
          reinterpret_cast<const DataType_*>(
              cross_attn_query_bias[i]->data<T>());
      // key
      params[i].cross_attention.key_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_key_weight[i]->data<T>());
      params[i].cross_attention.key_weight.bias =
          reinterpret_cast<const DataType_*>(cross_attn_key_bias[i]->data<T>());
      // value
      params[i].cross_attention.value_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_value_weight[i]->data<T>());
      params[i].cross_attention.value_weight.bias =
          reinterpret_cast<const DataType_*>(
              cross_attn_value_bias[i]->data<T>());
      // out proj
      params[i].cross_attention.attention_output_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_output_weight[i]->data<T>());
      params[i].cross_attention.attention_output_weight.bias =
          reinterpret_cast<const DataType_*>(
              cross_attn_output_bias[i]->data<T>());

      // ffn
      params[i].ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
          ffn_layernorm_weight[i]->data<T>());
      params[i].ffn_layernorm.beta =
          reinterpret_cast<const DataType_*>(ffn_layernorm_bias[i]->data<T>());
      // intermediate proj
      params[i].ffn.intermediate_weight.kernel =
          reinterpret_cast<const DataType_*>(
              ffn_intermediate_weight[i]->data<T>());
      params[i].ffn.intermediate_weight.bias =
          reinterpret_cast<const DataType_*>(
              ffn_intermediate_bias[i]->data<T>());
      // out proj
      params[i].ffn.output_weight.kernel =
          reinterpret_cast<const DataType_*>(ffn_output_weight[i]->data<T>());
      params[i].ffn.output_weight.bias =
          reinterpret_cast<const DataType_*>(ffn_output_bias[i]->data<T>());
    }

    decoding_params.layernorm.gamma =
        reinterpret_cast<const DataType_*>(decoder_layernorm_weight->data<T>());
    decoding_params.layernorm.beta =
        reinterpret_cast<const DataType_*>(decoder_layernorm_bias->data<T>());
    // for embedding
    decoding_params.embedding_table =
        reinterpret_cast<const DataType_*>(word_emb->data<T>());

    // for weight sharing matmul
    decoding_params.embedding_kernel =
        reinterpret_cast<const DataType_*>(embedding_weight->data<T>());
    // NOTE: the data type of the embedding bias for logits is different
    // between decoding with beam search and top-k/top-p sampling in
    // Faster Transformer when using float16.
    if ("beam_search" == decoding_strategy) {
      // for matmul bias
      decoding_params.embedding_bias =
          (embedding_bias)
              ? reinterpret_cast<const float*>(embedding_bias->data<float>())
              : nullptr;
    } else if ("topk_sampling" == decoding_strategy ||
               "topp_sampling" == decoding_strategy) {
      decoding_params.embedding_bias_T =
          (embedding_bias)
              ? reinterpret_cast<const DataType_*>(embedding_bias->data<T>())
              : nullptr;
    }
    decoding_params.position_encoding_table =
        reinterpret_cast<const DataType_*>(position_encoding_table->data<T>());

    if ("beam_search" == decoding_strategy) {
      DecodingBeamsearch<DecodingTraits_::OpType>* decoding_beamsearch_;
      decoding_beamsearch_ = new DecodingBeamsearch<DecodingTraits_::OpType>(
          allocator_,
          batch_size_,
          beam_width_,
          max_seq_len_,
          head_num_,
          size_per_head_,
          vocab_size,
          num_layer_,
          memory_hidden_dim,
          memory_max_seq_len,
          start_id_,
          end_id_,
          beam_search_diversity_rate_);

      decoding_beamsearch_->forward(params, decoding_params);

      delete decoding_beamsearch_;
    } else if ("topk_sampling" == decoding_strategy ||
               "topp_sampling" == decoding_strategy) {
      DecodingSampling<DecodingTraits_::OpType>* decoding_sampling_;
      decoding_sampling_ =
          new DecodingSampling<DecodingTraits_::OpType>(allocator_,
                                                        batch_size_,
                                                        max_seq_len_,
                                                        head_num_,
                                                        size_per_head_,
                                                        vocab_size,
                                                        num_layer_,
                                                        memory_hidden_dim,
                                                        memory_max_seq_len,
                                                        start_id_,
                                                        end_id_,
                                                        candidate_num_,
                                                        probability_threshold_);

      decoding_sampling_->forward(params, decoding_params);

      delete decoding_sampling_;
    } else {
      PADDLE_THROW(
          "Only beam_search, topk_sampling and topp_sampling are supported for "
          "Faster Transformer. ");
    }
    delete[] params;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plf = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    fusion_decoding,
    ops::FusionDecodingKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FusionDecodingKernel<paddle::platform::CUDADeviceContext,
                              plf::float16>);
