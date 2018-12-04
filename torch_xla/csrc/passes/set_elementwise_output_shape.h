#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Set the output shape of aten::{add, mul}.
void SetElementwiseOutputShape(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
