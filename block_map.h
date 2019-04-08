#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_

#include <cstdint>

namespace ruy {

static constexpr int kCacheFriendlyLoopThreshold = 32 * 1024;

enum class BlockMapTraversalOrder { kLinear, kFractalZ, kFractalU };

struct BlockMap {
  BlockMapTraversalOrder traversal_order;
  int rows;
  int cols;
  int num_blocks_base_log2;
  int rows_rectangularness_log2;
  int cols_rectangularness_log2;
  int kernel_rows;
  int kernel_cols;
  std::uint16_t smallr;
  std::uint16_t smallc;
  std::uint16_t missr;
  std::uint16_t missc;
};

void GetBlockByIndex(const BlockMap& block_map, std::uint32_t index,
                     std::uint16_t* block_r, std::uint16_t* block_c);

void MakeBlockMap(int rows, int cols, int depth, int kernel_rows,
                  int kernel_cols, int lhs_scalar_size, int rhs_scalar_size,
                  BlockMap* block_map);

void GetBlockMatrixCoords(const BlockMap& block_map, std::uint16_t block_r,
                          std::uint16_t block_c, int* start_r, int* start_c,
                          int* end_r, int* end_c);

inline std::uint16_t NumBlocksOfRows(const BlockMap& block_map) {
  return 1 << (block_map.num_blocks_base_log2 +
               block_map.rows_rectangularness_log2);
}

inline std::uint16_t NumBlocksOfCols(const BlockMap& block_map) {
  return 1 << (block_map.num_blocks_base_log2 +
               block_map.cols_rectangularness_log2);
}

inline std::uint32_t NumBlocks(const BlockMap& block_map) {
  return 1 << (2 * block_map.num_blocks_base_log2 +
               block_map.rows_rectangularness_log2 +
               block_map.cols_rectangularness_log2);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_BLOCK_MAP_H_
