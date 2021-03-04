#include <string>
#include <vector>

#include "fastertransformer/common.h"
#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/open_decoder.h"

#include "fusion_decoding_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {

class FusionDecodingOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto input_dims = ctx->GetInputDim("Input");
    auto beam_size = ctx->Attrs().Get<int>("beam_size");
    auto max_len = ctx->Attrs().Get<int64_t>("max_len");
    int batch_size = input_dims[0] / beam_size;

    auto output_dims = framework::make_ddim({max_len, batch_size, beam_size});
    ctx->SetOutputDim("OutputIds", output_dims);
    ctx->SetOutputDim("ParentIds", output_dims);
    ctx->SetOutputDim("SequenceLength",
                      framework::make_ddim({batch_size * beam_size}));
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FusionDecodingOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    // do op maker.
    // Add Parameters.
    AddInput("Input", "The input of fusion_decoding op. ");
    AddInput("MemSeqLen", "The sequence lengths of memory sequence. ");
    AddInput("WordEmbedding",
             "The input represents embedding tensors for target Ids. ");

    AddInput("SelfLayernormWeight",
             "The tensors of layer norm's scale before self "
             "attention layers. ")
        .AsDuplicable();
    AddInput("SelfLayernormBias",
             "The tensors of layer norm's bias before self attention "
             "layers. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("SelfQueryWeight",
             "The tensors of self attention's query projection weights. ")
        .AsDuplicable();
    AddInput("SelfQueryBias",
             "The tensors of self attention's query projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("SelfKeyWeight",
             "The tensors of self attention's key projection weights. ")
        .AsDuplicable();
    AddInput("SelfKeyBias",
             "The tensors of self attention's key projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("SelfValueWeight",
             "The tensors of self attention's value projection weights. ")
        .AsDuplicable();
    AddInput("SelfValueBias",
             "The tensors of self attention's value projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("SelfOutWeight",
             "The tensors of self attention's output projection weights. ")
        .AsDuplicable();
    AddInput("SelfOutBias",
             "The tensors of self attention's output projection biases. ")
        .AsDuplicable()
        .AsDispensable();

    AddInput(
        "CrossLayernormWeight",
        "The tensors of layer norm's weights before cross attention layers. ")
        .AsDuplicable();
    AddInput(
        "CrossLayernormBias",
        "The tensors of layer norm's biases before cross attention layers. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("CrossQueryWeight",
             "The tensors of cross attention's query projection weights. ")
        .AsDuplicable();
    AddInput("CrossQueryBias",
             "The tensors of cross attention's query projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("CrossKeyWeight",
             "The tensors of cross attention's key projection weights. ")
        .AsDuplicable();
    AddInput("CrossKeyBias",
             "The tensors of cross attention's key projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("CrossValueWeight",
             "The tensors of cross attention's value projection weights. ")
        .AsDuplicable();
    AddInput("CrossValueBias",
             "The tensors of cross attention's value projection biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("CrossOutWeight",
             "The tensors of cross attention's output projection weights. ")
        .AsDuplicable();
    AddInput("CrossOutBias",
             "The tensors of cross attention's output projection biases. ")
        .AsDuplicable()
        .AsDispensable();

    AddInput("FFNLayernormWeight",
             "The tensors of layer norm's weights before ffn. ")
        .AsDuplicable();
    AddInput("FFNLayernormBias",
             "The tensors of layer norm's biases before ffn. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("FFNInterWeight", "The tensors of inter fc weights. ")
        .AsDuplicable();
    AddInput("FFNInterBias", "The tensors of inter fc biases. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("FFNOutWeight", "The tensors of output weights. ").AsDuplicable();
    AddInput("FFNOutBias", "The tensors of output biases. ")
        .AsDuplicable()
        .AsDispensable();

    AddInput("DecoderLayernormWeight",
             "The tensor of layer norm's weights after decoders. ");
    AddInput("DecoderLayernormBias",
             "The tensor of layer norm's biases after decoders. ");
    AddInput("EmbWeight", "The tensor of logits projection weight. ");
    AddInput("EmbBias", "The tensor of logits projection bias. ")
        .AsDispensable();
    AddInput("PositionEncEmb", "The tensor of positional enbedding table. ");

    AddOutput("OutputIds", "The tensor of output ids. ");
    AddOutput("ParentIds", "The tensor of parent ids. ");
    AddOutput("SequenceLength", "The tensor of sequence length. ");

    AddAttr<std::string>(
        "decoding_strategy",
        "Decoding strategies. As for now, only beam search is supported. ")
        .SetDefault("beam_search");
    AddAttr<int>("beam_size", "The beam size for beam search. ").SetDefault(1);
    AddAttr<int>("n_head", "The number of heads. ").SetDefault(8);
    AddAttr<int>("size_per_head", "The size per head. ").SetDefault(64);
    AddAttr<int>("num_layer", "The number of layers. ").SetDefault(6);
    AddAttr<int>("bos_id", "Start id. ").SetDefault(0);
    AddAttr<int>("eos_id", "End id. ").SetDefault(1);
    AddAttr<int64_t>("max_len", "Max output length. ").SetDefault(256);
    AddAttr<float>("beam_search_diversity_rate",
                   "The diversity rate for beam search. ")
        .SetDefault(0.0);

    AddComment(R"DOC(
    Decoding Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(fusion_decoding,
                             ops::FusionDecodingOp,
                             ops::FusionDecodingOpMaker);
REGISTER_OP_CPU_KERNEL(fusion_decoding, ops::NotImpleKernel<float>);
