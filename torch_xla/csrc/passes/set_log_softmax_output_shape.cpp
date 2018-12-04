#include "set_log_softmax_output_shape.h"

namespace torch {
namespace jit {

namespace {

void SetLogSoftmaxOutputShape(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      SetLogSoftmaxOutputShape(sub_block);
    }
    if (it->kind() == aten::log_softmax) {
      const auto lhs_type = it->input(0)->type()->cast<CompleteTensorType>();
      if (lhs_type) {
        it->output()->setType(CompleteTensorType::create(
            lhs_type->scalarType(), lhs_type->device(), lhs_type->sizes()));
      }
    }
  }
}

}  // namespace

void SetLogSoftmaxOutputShape(const std::shared_ptr<Graph>& graph) {
  SetLogSoftmaxOutputShape(graph->block());
}

}  // namespace jit
}  // namespace torch
