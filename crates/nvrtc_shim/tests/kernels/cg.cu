#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
extern "C" __global__ void cg_sum(const float* in, float* out) {
  cg::thread_block blk = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(blk);
  float v = in[blk.thread_rank()];
  float s = cg::reduce(tile, v, cg::plus<float>());
  if (tile.thread_rank() == 0) out[blk.thread_rank() / 32] = s;
}
