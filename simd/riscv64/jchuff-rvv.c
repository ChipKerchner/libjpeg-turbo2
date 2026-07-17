/*
 * Huffman Encoding (RISCV64)
 *
 * Copyright (C) 2020, 2022, 2024-2026, D. R. Commander.
 * Copyright (C) 2026 Chip Kerchner.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * NOTE: All referenced figures are from
 * Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
 */

#ifndef HUFFMAN_ENCODER_RVV
#define HUFFMAN_ENCODER_RVV     jsimd_huff_encode_one_block_zbb_rvv
#endif

#ifdef USE_HUFFMAN_ENCODER_RVV_256
#include "../jsimdint.h"
#include <riscv_vector.h>
#include "jchuff.h"

#include <float.h>
#include <limits.h>


#ifndef HUFFMAN_ENCODER_256_RVV
#define HUFFMAN_ENCODER_256_RVV jsimd_huff_encode_one_block_zbb_256_rvv
#endif

#define VEC_LEN     (DCTSIZE * 4)

#define CNT_BIAS    (127 - 1)

static const uint8_t jsimd_huff_encode_one_block_consts[] = {
    0,   2,  16,  32,  18,   4,   6,  20,
   34,  48,  64,  50,  36,  22,   8,  10,
   24,  38,  52,  66,  80,  96,  82,  68,
   54,  40,  26,  12,  14,  28,  42,  56,
   70,  84,  98, 112, 114, 100,  86,  72,
   58,  44,  30,  46,  60,  74,  88, 102,
  116, 118, 104,  90,  76,  62,  78,  92,
  106, 120, 122, 108,  94, 110, 124, 126
};
#else
#undef VEC_CLZ
#undef VEC_LEN

#define VEC_LEN     (DCTSIZE * 2)
#endif

/* Creates a vector of out = 16 - ctz(abs(in)) */
#ifdef __riscv_zvbb
#define VEC_CLZ(in, out, mask, zero, shift) \
  { \
    mask = __riscv_vmslt_vx_i16m2_b8(in, 0, VEC_LEN); \
    shift = __riscv_vclz_v_u16m2(__riscv_vreinterpret_v_i16m2_u16m2( \
      __riscv_vneg_v_i16m2_mu(mask, in, in, VEC_LEN)), VEC_LEN); \
    out = __riscv_vrsub_vx_u16m2(shift, 16, VEC_LEN); \
  }
#else
#define VEC_CLZ(in, out, mask, zero, shift) \
  { \
    mask = __riscv_vmslt_vx_i16m2_b8(in, 0, VEC_LEN); \
    out = __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4( \
      __riscv_vfwcvt_f_x_v_f32m4(__riscv_vneg_v_i16m2_mu(mask, in, in, \
      VEC_LEN), VEC_LEN)), FLT_MANT_DIG - 1, VEC_LEN); \
    out = __riscv_vsub_vx_u16m2_mu(zero, out, out, CNT_BIAS, VEC_LEN); \
    shift = __riscv_vrsub_vx_u16m2(out, 16, VEC_LEN); \
  }
#endif

HIDDEN JOCTET *
HUFFMAN_ENCODER_RVV(void *state, JOCTET *buffer, JCOEFPTR block,
                    int last_dc_val, void *dctbl, void *actbl)
{
  uint16_t block_diff[DCTSIZE2];

  /* Load lookup table indices for rows of zig-zag ordering. */
  const vuint8m1_t idx_rows0 =
    __riscv_vle8_v_u8m1(jsimd_huff_encode_one_block_consts + (VEC_LEN * 0),
    VEC_LEN);
  const vuint8m1_t idx_rows1 =
    __riscv_vle8_v_u8m1(jsimd_huff_encode_one_block_consts + (VEC_LEN * 1),
    VEC_LEN);
#if VEC_LEN == 16
  const vuint8m1_t idx_rows2 =
    __riscv_vle8_v_u8m1(jsimd_huff_encode_one_block_consts + (VEC_LEN * 2),
    VEC_LEN);
  const vuint8m1_t idx_rows3 =
    __riscv_vle8_v_u8m1(jsimd_huff_encode_one_block_consts + (VEC_LEN * 3),
    VEC_LEN);
#endif

  /* Shuffle coefficients into zig-zag order. */
  vint16m2_t rows0 =
    __riscv_vluxei8_v_i16m2((int16_t *)(block), idx_rows0, VEC_LEN);
  vint16m2_t rows1 =
    __riscv_vluxei8_v_i16m2((int16_t *)(block), idx_rows1, VEC_LEN);
#if VEC_LEN == 16
  vint16m2_t rows2 =
    __riscv_vluxei8_v_i16m2((int16_t *)(block), idx_rows2, VEC_LEN);
  vint16m2_t rows3 =
    __riscv_vluxei8_v_i16m2((int16_t *)(block), idx_rows3, VEC_LEN);
#endif

  /* DCT block is now in zig-zag order; start Huffman encoding process. */

  /* Construct bitmap to accelerate encoding of AC coefficients.  A set bit
   * means that the corresponding coefficient != 0.
   */
  vbool8_t rows_mask0 = __riscv_vmsne_vx_i16m2_b8(rows0, 0, VEC_LEN);
  vbool8_t rows_mask1 = __riscv_vmsne_vx_i16m2_b8(rows1, 0, VEC_LEN);
#if VEC_LEN == 16
  vbool8_t rows_mask2 = __riscv_vmsne_vx_i16m2_b8(rows2, 0, VEC_LEN);
  vbool8_t rows_mask3 = __riscv_vmsne_vx_i16m2_b8(rows3, 0, VEC_LEN);
  uint16_t bitmap0 = __riscv_vmv_x_s_u16m1_u16(
    __riscv_vreinterpret_v_b8_u16m1(rows_mask0));
  uint16_t bitmap1 = __riscv_vmv_x_s_u16m1_u16(
    __riscv_vreinterpret_v_b8_u16m1(rows_mask1));
  uint16_t bitmap2 = __riscv_vmv_x_s_u16m1_u16(
    __riscv_vreinterpret_v_b8_u16m1(rows_mask2));
  uint16_t bitmap3 = __riscv_vmv_x_s_u16m1_u16(
    __riscv_vreinterpret_v_b8_u16m1(rows_mask3));
  /* Shift right to remove DC bit. */
  uint64_t bitmap = (uint64_t)(bitmap0 >> 1) |
    ((uint64_t)(bitmap1) << ((VEC_LEN * 1) - 1)) |
    ((uint64_t)(bitmap2) << ((VEC_LEN * 2) - 1)) |
    ((uint64_t)(bitmap3) << ((VEC_LEN * 3) - 1));
#else
  uint32_t bitmap0 = __riscv_vmv_x_s_u32m1_u32(
    __riscv_vreinterpret_v_b8_u32m1(rows_mask0));
  uint32_t bitmap1 = __riscv_vmv_x_s_u32m1_u32(
    __riscv_vreinterpret_v_b8_u32m1(rows_mask1));
  /* Shift right to remove DC bit. */
  uint64_t bitmap = (uint64_t)(bitmap0 >> 1) |
    ((uint64_t)(bitmap1) << ((VEC_LEN * 1) - 1));
#endif
  /* Count bits set (number of non-zero coefficients) in bitmap. */
  size_t non_zero_coefficients = BUILTIN_POPCNTL(bitmap);

  /* Set up state and bit buffer for output bitstream. */
  working_state *state_ptr = (working_state *)state;
  int free_bits = state_ptr->cur.free_bits;
  size_t put_buffer = state_ptr->cur.put_buffer;

  /* Encode DC coefficient. */

  /* Compute DC coefficient difference value (F.1.1.5.1). */
  int64_t diff = block[0] - last_dc_val;
  /* For negative coeffs: diff = abs(coeff) - 1 = ~abs(coeff) */
  int64_t mask = diff >> ((sizeof(uint64_t) * CHAR_BIT) - 1);
  diff += mask;
  int64_t abs_diff = diff ^ mask;
  uint64_t lz = BUILTIN_CLZL(abs_diff);
  uint64_t nbits = (sizeof(uint64_t) * CHAR_BIT) - lz;
  diff = ((uint64_t)(diff) << lz) >> lz;
  /* Emit Huffman-coded symbol and additional diff bits. */
  PUT_CODE(((c_derived_tbl *)dctbl)->ehufco[nbits],
           ((c_derived_tbl *)dctbl)->ehufsi[nbits], diff)

  /* Encode AC coefficients. */

  uint64_t r = 0;  /* r = run length of zeros */
  uint64_t i = 1;  /* i = number of coefficients encoded */
  /* Code and size information for a run length of 16 zero coefficients */
  const unsigned int code_0xf0 = ((c_derived_tbl *)actbl)->ehufco[0xf0];
  const unsigned int size_0xf0 = ((c_derived_tbl *)actbl)->ehufsi[0xf0];

  /* The most efficient method of computing nbits and diff depends on the
   * number of non-zero coefficients.  If the bitmap is not too sparse (> 8
   * non-zero AC coefficients), it is beneficial to do all of the work using
   * RVV else we do some of the work using RVV and the rest on demand using
   * scalar code.
   */
  if (non_zero_coefficients > 8) {
    uint16_t block_nbits[DCTSIZE2];
    vuint16m2_t out0, out1, shift0, shift1;
    vbool8_t mask0, mask1;

    /* Compute nbits needed to specify magnitude of each coefficient. */
    VEC_CLZ(rows0, out0, mask0, rows_mask0, shift0)
    VEC_CLZ(rows1, out1, mask1, rows_mask1, shift1)
    /* Store nbits. */
    __riscv_vse16_v_u16m2(block_nbits + (VEC_LEN * 0), out0, VEC_LEN);
    __riscv_vse16_v_u16m2(block_nbits + (VEC_LEN * 1), out1, VEC_LEN);
    /* Mask bits not required to specify sign and amplitude of diff. */
    rows0 = __riscv_vsub_vx_i16m2_mu(mask0, rows0, rows0, 1, VEC_LEN);
    rows1 = __riscv_vsub_vx_i16m2_mu(mask1, rows1, rows1, 1, VEC_LEN);
    rows0 = __riscv_vsll_vv_i16m2(rows0, shift0, VEC_LEN);
    rows1 = __riscv_vsll_vv_i16m2(rows1, shift1, VEC_LEN);
    rows0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsrl_vv_u16m2(
       __riscv_vreinterpret_v_i16m2_u16m2(rows0), shift0, VEC_LEN));
    rows1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsrl_vv_u16m2(
       __riscv_vreinterpret_v_i16m2_u16m2(rows1), shift1, VEC_LEN));
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 0), rows0, VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 1), rows1, VEC_LEN);
#if VEC_LEN == 16
    vuint16m2_t out2, out3, shift2, shift3;
    vbool8_t mask2, mask3;
    VEC_CLZ(rows2, out2, mask2, rows_mask2, shift2)
    VEC_CLZ(rows3, out3, mask3, rows_mask3, shift3)
    __riscv_vse16_v_u16m2(block_nbits + (VEC_LEN * 2), out2, VEC_LEN);
    __riscv_vse16_v_u16m2(block_nbits + (VEC_LEN * 3), out3, VEC_LEN);
    rows2 = __riscv_vsub_vx_i16m2_mu(mask2, rows2, rows2, 1, VEC_LEN);
    rows3 = __riscv_vsub_vx_i16m2_mu(mask3, rows3, rows3, 1, VEC_LEN);
    rows2 = __riscv_vsll_vv_i16m2(rows2, shift2, VEC_LEN);
    rows3 = __riscv_vsll_vv_i16m2(rows3, shift3, VEC_LEN);
    rows2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsrl_vv_u16m2(
       __riscv_vreinterpret_v_i16m2_u16m2(rows2), shift2, VEC_LEN));
    rows3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsrl_vv_u16m2(
       __riscv_vreinterpret_v_i16m2_u16m2(rows3), shift3, VEC_LEN));
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 2), rows2, VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 3), rows3, VEC_LEN);
#endif

    while (bitmap != 0) {
      r = BUILTIN_CTZL(bitmap);
      i += r;
      bitmap >>= (r + 1);
      nbits = block_nbits[i];
      diff = block_diff[i];
      while (r >= 16) {
        /* If run length >= 16, emit special run-length-16 codes. */
        PUT_BITS(code_0xf0, size_0xf0)
        r -= 16;
      }
      /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
      uint64_t rs = (r << 4) + nbits;
      PUT_CODE(((c_derived_tbl *)actbl)->ehufco[rs],
               ((c_derived_tbl *)actbl)->ehufsi[rs], diff)
      i++;
    }
  } else if (bitmap != 0) {
    uint16_t block_abs[DCTSIZE2];
    /* Compute and store absolute value of coefficients. */
    vbool8_t mask0 = __riscv_vmslt_vx_i16m2_b8(rows0, 0, VEC_LEN);
    vbool8_t mask1 = __riscv_vmslt_vx_i16m2_b8(rows1, 0, VEC_LEN);
    vint16m2_t abs_rows0 = __riscv_vneg_v_i16m2_mu(mask0, rows0, rows0,
      VEC_LEN);
    vint16m2_t abs_rows1 = __riscv_vneg_v_i16m2_mu(mask1, rows1, rows1,
      VEC_LEN);
    /* Compute diff bits (without nbits mask) and store. */
    rows0 = __riscv_vsub_vx_i16m2_mu(mask0, rows0, rows0, 1, VEC_LEN);
    rows1 = __riscv_vsub_vx_i16m2_mu(mask1, rows1, rows1, 1, VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_abs) + (VEC_LEN * 0), abs_rows0,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_abs) + (VEC_LEN * 1), abs_rows1,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 0), rows0,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 1), rows1,
      VEC_LEN);
#if VEC_LEN == 16
    vbool8_t mask2 = __riscv_vmslt_vx_i16m2_b8(rows2, 0, VEC_LEN);
    vbool8_t mask3 = __riscv_vmslt_vx_i16m2_b8(rows3, 0, VEC_LEN);
    vint16m2_t abs_rows2 = __riscv_vneg_v_i16m2_mu(mask2, rows2, rows2,
      VEC_LEN);
    vint16m2_t abs_rows3 = __riscv_vneg_v_i16m2_mu(mask3, rows3, rows3,
      VEC_LEN);
    rows2 = __riscv_vsub_vx_i16m2_mu(mask2, rows2, rows2, 1, VEC_LEN);
    rows3 = __riscv_vsub_vx_i16m2_mu(mask3, rows3, rows3, 1, VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_abs) + (VEC_LEN * 2), abs_rows2,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_abs) + (VEC_LEN * 3), abs_rows3,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 2), rows2,
      VEC_LEN);
    __riscv_vse16_v_i16m2((int16_t *)(block_diff) + (VEC_LEN * 3), rows3,
      VEC_LEN);
#endif

    /* Same as above but must mask diff bits and compute nbits on demand. */
    while (bitmap != 0) {
      r = BUILTIN_CTZL(bitmap);
      i += r;
      bitmap >>= (r + 1);
      lz = BUILTIN_CLZL((uint64_t)(block_abs[i]));
      nbits = (sizeof(uint64_t) * CHAR_BIT) - lz;
      diff = ((uint64_t)(block_diff[i]) << lz) >> lz;
      while (r >= 16) {
        /* If run length >= 16, emit special run-length-16 codes. */
        PUT_BITS(code_0xf0, size_0xf0)
        r -= 16;
      }
      /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
      uint64_t rs = (r << 4) + nbits;
      PUT_CODE(((c_derived_tbl *)actbl)->ehufco[rs],
               ((c_derived_tbl *)actbl)->ehufsi[rs], diff)
      i++;
    }
  }

  /* If the last coefficient(s) were zero, emit an end-of-block (EOB) code.
   * The value of RS for the EOB code is 0.
   */
  if (i != 64) {
    PUT_BITS(((c_derived_tbl *)actbl)->ehufco[0],
             ((c_derived_tbl *)actbl)->ehufsi[0])
  }

  state_ptr->cur.put_buffer = put_buffer;
  state_ptr->cur.free_bits = free_bits;

  return buffer;
}

#undef USE_HUFFMAN_ENCODER_RVV_256
#undef HUFFMAN_ENCODER_RVV
