#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

// Set the output shape of aten::{threshold, relu}.
void SetThresholdOutputShape(const std::shared_ptr<Graph>& graph);

}  // namespace jit
}  // namespace torch
