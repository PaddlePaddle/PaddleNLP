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
             "The tensors of layer norm's scale before different self "
             "attention layers. ")
        .AsDuplicable();
    AddInput("SelfLayernormBias",
             "The tensors of layer norm's bias before different self attention "
             "layers. ")
        .AsDuplicable()
        .AsDispensable();
    AddInput("SelfQueryWeight",
             "The tensors of self attention's query projection weights. ")
        .AsDuplicable();
    AddInput("SelfQueryBias", "").AsDuplicable().AsDispensable();
    AddInput("SelfKeyWeight", "").AsDuplicable();
    AddInput("SelfKeyBias", "").AsDuplicable().AsDispensable();
    AddInput("SelfValueWeight", "").AsDuplicable();
    AddInput("SelfValueBias", "").AsDuplicable().AsDispensable();
    AddInput("SelfOutWeight", "").AsDuplicable();
    AddInput("SelfOutBias", "").AsDuplicable().AsDispensable();

    AddInput("CrossLayernormWeight", "").AsDuplicable();
    AddInput("CrossLayernormBias", "").AsDuplicable().AsDispensable();
    AddInput("CrossQueryWeight", "").AsDuplicable();
    AddInput("CrossQueryBias", "").AsDuplicable().AsDispensable();
    AddInput("CrossKeyWeight", "").AsDuplicable();
    AddInput("CrossKeyBias", "").AsDuplicable().AsDispensable();
    AddInput("CrossValueWeight", "").AsDuplicable();
    AddInput("CrossValueBias", "").AsDuplicable().AsDispensable();
    AddInput("CrossOutWeight", "").AsDuplicable();
    AddInput("CrossOutBias", "").AsDuplicable().AsDispensable();

    AddInput("FFNLayernormWeight", "").AsDuplicable();
    AddInput("FFNLayernormBias", "").AsDuplicable().AsDispensable();
    AddInput("FFNInterWeight", "").AsDuplicable();
    AddInput("FFNInterBias", "").AsDuplicable().AsDispensable();
    AddInput("FFNOutWeight", "").AsDuplicable();
    AddInput("FFNOutBias", "").AsDuplicable().AsDispensable();

    AddInput("DecoderLayernormWeight", "");
    AddInput("DecoderLayernormBias", "");
    AddInput("EmbWeight", "");
    AddInput("EmbBias", "").AsDispensable();
    AddInput("PositionEncEmb", "");

    AddOutput("OutputIds", "");
    AddOutput("ParentIds", "");
    AddOutput("SequenceLength", "");

    AddAttr<std::string>("decoding_strategy", "").SetDefault("beam_search");
    AddAttr<int>("beam_size", "").SetDefault(1);
    AddAttr<int>("n_head", "").SetDefault(8);
    AddAttr<int>("size_per_head", "").SetDefault(64);
    AddAttr<int>("num_layer", "").SetDefault(6);
    AddAttr<int>("bos_id", "").SetDefault(0);
    AddAttr<int>("eos_id", "").SetDefault(1);
    AddAttr<int64_t>("max_len", "").SetDefault(256);
    AddAttr<float>("beam_search_diversity_rate", "").SetDefault(0.0);

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
