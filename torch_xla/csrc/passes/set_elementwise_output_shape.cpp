#include "set_mat_mul_output_shape.h"

namespace torch {
namespace jit {

namespace {

void SetElementwiseOutputShape(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      SetElementwiseOutputShape(sub_block);
    }
    if (it->kind() == aten::add || it->kind() == aten::mul) {
      const auto lhs_type = it->input(0)->type()->cast<CompleteTensorType>();
      const auto output_type =
          it->output(0)->type()->cast<CompleteTensorType>();
      if (lhs_type && !output_type) {
        it->output()->setType(CompleteTensorType::create(
            lhs_type->scalarType(), lhs_type->device(), lhs_type->sizes()));
      }
    }
  }
}

}  // namespace

void SetElementwiseOutputShape(const std::shared_ptr<Graph>& graph) {
  SetElementwiseOutputShape(graph->block());
}

}  // namespace jit
}  // namespace torch
