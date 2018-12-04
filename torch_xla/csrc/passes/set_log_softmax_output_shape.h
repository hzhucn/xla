#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Set the output shape of log_softmax.
void SetLogSoftmaxOutputShape(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
