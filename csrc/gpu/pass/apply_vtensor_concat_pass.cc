#include "paddle/extension.h"
#include <iostream>

namespace {

class ApplyVTensorConcatPattern : public paddle::drr::DrrPatternBase {
public:
  std::string name() const override { return "ApplyVTensorConcatPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &full = pat.Op("pd_op.full", {{"value", pat.Attr("axis")}});
    pat.Tensor("concat_axis") = full();
    
    const auto &concat_combine = pat.Op("builtin.combine");
    pat.Tensor("concat_in") = concat_combine(pat.Tensor("concat_in1"), pat.Tensor("concat_in2"));

    const auto &concat = pat.Op("pd_op.concat");
    pat.Tensor("concat_out") = concat(pat.Tensor("concat_in"), pat.Tensor("concat_axis"));

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      auto shape1 = pir::GetShapeFromValue(match_ctx.Tensor("concat_in1"));
      auto shape2 = pir::GetShapeFromValue(match_ctx.Tensor("concat_in2"));
      auto matched = (
        match_ctx.Attr<double>("axis") == 1.0 &&
        shape1.size() == 4 &&
        shape2.size() == 4 &&
        true
      );

      if (matched) {
        bool has_yield = false, has_attn = false;
        auto out = match_ctx.Tensor("concat_out");
        for (auto op = out.use_begin(); op != out.use_end(); ++op) {
          auto name = op->owner()->name();
          // std::cout << name << " ";
          has_yield |= name == "cf.yield";
          has_attn |= name == "pd_op.flash_attn";
        }
        // std::cout << std::endl;
        matched &= has_yield;
        // matched &= has_attn;
      }
      
      return matched;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &vconcat = res.Op(
      "custom_op.vtensor_reserve_one_token",
      {{"transposed_input", res.BoolAttr(false)}}
    );
    res.Tensor("concat_out") = vconcat(res.Tensor("concat_in1"), res.Tensor("concat_in2"));
  }
};

class ApplyVTensorConcatPass : public pir::PatternRewritePass {
public:
  ApplyVTensorConcatPass() : pir::PatternRewritePass("apply_vtensor_concat_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ApplyVTensorConcatPattern>(context));
    return ps;
  }
};

} // namespace

REGISTER_IR_PASS(apply_vtensor_concat_pass, ApplyVTensorConcatPass);