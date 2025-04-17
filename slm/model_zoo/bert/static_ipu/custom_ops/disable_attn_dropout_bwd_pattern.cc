/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/softmax.hpp>
#include <popart/logging.hpp>
#include <popart/opidentifier.hpp>

#include "utils.cc"

// Tests have found that disabling dropout in the backwards pass of the attention, just before the softmax,
// can improve SQuAD fine-tuning. This pattern finds that op replaces it with an identity op.
class DisableAttnDropoutBwdPattern : public popart::PreAliasPattern {
public:
  bool matches(popart::Op *op) const override {
    int check_levels = 2;

    if (!op->isConvertibleTo<popart::DropoutGradOp>()) {
      return false;
    }

    // Is dropout enabled? If ratio is 0, we don't need to apply the pattern.
    auto dropoutGradOp = dynamic_cast<popart::DropoutGradOp*>(op);
    if (dropoutGradOp->getRatio() == 0.f) {
      return false;
    }

    // The specific attention DropoutGradOp we want to cull sits between a matmul and a softmax,
    // so we'll look through producers and consumers and see if we can find them.
    auto grad = op->input->tensor(popart::DropoutGradOp::getGradInIndex());

    // The MatMulPattern converts the MatMulLhsGradOp to a MatMulOp
    // There doesn't seem to be a way to check if a pattern is enabled from inside another pattern.
    // The IR holds the patterns object, but it’s inaccessible for checking the status of individual patterns.
    // Check both, with the most likely first.
    bool hasMatMulProducer = search_producers_for<popart::MatMulOp>(grad, check_levels) != nullptr;
    if (!hasMatMulProducer) {
      hasMatMulProducer |= search_producers_for<popart::MatMulLhsGradOp>(grad, check_levels) != nullptr;
    }

    return hasMatMulProducer && search_consumers_for<popart::SoftmaxGradOp>(grad) != nullptr;
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

  bool apply(popart::Op *op) const override {
    if (!op->isConvertibleTo<popart::DropoutGradOp>()) {
      return false;
    }

    auto dropoutGradOp = dynamic_cast<popart::DropoutGradOp*>(op);
    auto identityOp = makeReplacementOpInIr(popart::Onnx::Operators::Identity_1,
                                            dropoutGradOp,
                                            "");

    auto inputId = dropoutGradOp->inId(popart::DropoutGradOp::getGradInIndex());
    auto outputId = dropoutGradOp->outId(popart::DropoutGradOp::getOutIndex());
    dropoutGradOp->disconnectAllInputs();
    dropoutGradOp->disconnectAllOutputs();
    dropoutGradOp->getGraph().eraseOp(dropoutGradOp->id);

    identityOp->connectInTensor(popart::IdentityOp::getInIndex(), inputId);
    identityOp->connectOutTensor(popart::IdentityOp::getOutIndex(), outputId);
    identityOp->setup();

    return true;
  }
};


static popart::PatternCreator<DisableAttnDropoutBwdPattern> disableAttnDropoutBwdPatternCreator("DisableAttnDropoutBwdPattern", false);
