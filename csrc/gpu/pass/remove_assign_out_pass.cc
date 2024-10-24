#include "paddle/extension.h"
#include <iostream>
#include <algorithm>

namespace {

class RemoveAssignOutPattern : public paddle::drr::DrrPatternBase {
public:
  std::string name() const override { return "RemoveAssignOutPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &assign_out_ = pat.Op("pd_op.assign_out_");
    assign_out_({&pat.Tensor("assign_in"), &pat.Tensor("assign_out")}, {&pat.Tensor("out")});

    pat.AddConstraint([](const paddle::drr::MatchContext &match_ctx) {
      auto &out = match_ctx.Tensor("out");
      auto parent_block = out.defining_op()->GetParent();
      auto parent_op = out.defining_op()->GetParentOp();

      auto &assign_out = match_ctx.Tensor("assign_out");
      
      if (
        parent_block && parent_op &&
        parent_op->name() == "pd_op.while" &&
        out.use_count() == 1 && out.use_begin()->owner()->name() == "cf.yield" &&
        std::find(parent_block->args_begin(), parent_block->args_end(), assign_out) != parent_block->args_end()
      ) {
        return true;
      }
      return false;
    });

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