#define HUFFMAN_ENCODER_MCU_FIRST_256_RVV  jsimd_encode_mcu_AC_first_prepare_rvv_vlen256
#define HUFFMAN_ENCODER_MCU_REFINE_256_RVV jsimd_encode_mcu_AC_refine_prepare_rvv_vlen256

#define USE_HUFFMAN_ENCODER_MCU_RVV_256
#define HUFFMAN_ENCODER_MCU_FIRST_RVV      HUFFMAN_ENCODER_MCU_FIRST_256_RVV
#define HUFFMAN_ENCODER_MCU_REFINE_RVV     HUFFMAN_ENCODER_MCU_REFINE_256_RVV

#include "jcphuff-rvv.c"

#define HUFFMAN_ENCODER_MCU_FIRST_RVV      jsimd_encode_mcu_AC_first_prepare_rvv
#define HUFFMAN_ENCODER_MCU_REFINE_RVV     jsimd_encode_mcu_AC_refine_prepare_rvv

#include "jcphuff-rvv.c"
