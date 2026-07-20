#define HUFFMAN_ENCODER_256_RVV     jsimd_huff_encode_one_block_zvbb_rvv_vlen256

#define USE_HUFFMAN_ENCODER_RVV_256
#define HUFFMAN_ENCODER_RVV         HUFFMAN_ENCODER_256_RVV

#include "jchuff-rvv.c"

#define HUFFMAN_ENCODER_RVV         jsimd_huff_encode_one_block_zvbb_rvv

#include "jchuff-rvv.c"
