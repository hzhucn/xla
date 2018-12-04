#include "set_threshold_output_shape.h"

namespace torch {
namespace jit {

namespace {

void SetThresholdOutputShape(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      SetThresholdOutputShape(sub_block);
    }
    if (it->kind() == aten::threshold || it->kind() == aten::relu) {
      const auto lhs_type = it->input(0)->type()->cast<CompleteTensorType>();
      if (lhs_type) {
        it->output()->setType(CompleteTensorType::create(
            lhs_type->scalarType(), lhs_type->device(), lhs_type->sizes()));
      }
    }
  }
}

}  // namespace

void SetThresholdOutputShape(const std::shared_ptr<Graph>& graph) {
  SetThresholdOutputShape(graph->block());
}

}  // namespace jit
}  // namespace torch
