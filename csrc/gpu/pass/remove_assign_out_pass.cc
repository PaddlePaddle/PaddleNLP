#include "paddle/extension.h"
#include <iostream>

namespace {

class RemoveAssignOutPattern : public paddle::drr::DrrPatternBase {
public:
  std::string name() const override { return "RemoveAssignOutPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &assign_out_ = pat.Op("pd_op.assign_out_");
    assign_out_({&pat.Tensor("assign_in"), &pat.Tensor("assign_out")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("out").Assign(res.Tensor("assign_in"));
  }
};

class RemoveAssignOutPass : public pir::PatternRewritePass {
public:
  RemoveAssignOutPass() : pir::PatternRewritePass("remove_assign_out_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<RemoveAssignOutPattern>(context));
    return ps;
  }
};

} // namespace

REGISTER_IR_PASS(remove_assign_out_pass, RemoveAssignOutPass);