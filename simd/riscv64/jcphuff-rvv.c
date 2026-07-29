/*
 * Data Preparation for Progressive Huffman Encoding (RISCV64)
 *
 * Copyright (C) 2022, 2024-2025, D. R. Commander.
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
 */

#ifndef HUFFMAN_ENCODER_MCU_FIRST_RVV
#define HUFFMAN_ENCODER_MCU_FIRST_RVV      jsimd_encode_mcu_AC_first_prepare_rvv
#endif
#ifndef HUFFMAN_ENCODER_MCU_REFINE_RVV
#define HUFFMAN_ENCODER_MCU_REFINE_RVV     jsimd_encode_mcu_AC_refine_prepare_rvv
#endif

#ifdef USE_HUFFMAN_ENCODER_MCU_RVV_256
#include "../jsimdint.h"
#include <riscv_vector.h>
#include "jchuff.h"

#include <limits.h>


#ifndef HUFFMAN_ENCODER_MCU_FIRST_256_RVV
#define HUFFMAN_ENCODER_MCU_FIRST_256_RVV  jsimd_encode_mcu_AC_first_prepare_rvv_vlen256
#endif
#ifndef HUFFMAN_ENCODER_MCU_REFINE_256_RVV
#define HUFFMAN_ENCODER_MCU_REFINE_256_RVV jsimd_encode_mcu_AC_refine_prepare_rvv_vlen256
#endif
#endif


/* Data preparation for encode_mcu_AC_first().
 *
 * The equivalent scalar C function (encode_mcu_AC_first_prepare()) can be
 * found in jcphuff.c.
 */

HIDDEN void
HUFFMAN_ENCODER_MCU_FIRST_RVV(const JCOEF *block,
                              const int *jpeg_natural_order_start, int Sl,
                              int Al, UJCOEF *values,
                              size_t *zerobits)
{
#ifdef __riscv_zbb
  UJCOEF *values_ptr = values;
  UJCOEF *diff_values_ptr = values + DCTSIZE2;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint32m4_t off = __riscv_vle32_v_u32m4(
      (uint32_t *)(jpeg_natural_order_start), 16);
    off = __riscv_vsll_vx_u32m4(off, 1, 16);
    vint16m2_t coefs = __riscv_vluxei32_v_i16m2(block, off, 16);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool8_t mask = __riscv_vmslt_vx_i16m2_b8(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m2_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m2(coefs, Al, 16);

    /* Compute diff values. */
    vint16m2_t diffs = __riscv_vnot_v_i16m2_mu(mask, coefs, coefs, 16);

    /* Store transformed coefficients and diff values. */
    __riscv_vse16_v_u16m2(values_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(coefs), 16);
    __riscv_vse16_v_u16m2(diff_values_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(diffs), 16);
#else
    vuint32m2_t off = __riscv_vle32_v_u32m2(
      (uint32_t *)(jpeg_natural_order_start), 16);
    off = __riscv_vsll_vx_u32m2(off, 1, 16);
    vint16m1_t coefs = __riscv_vluxei32_v_i16m1(block, off, 16);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m1_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m1(coefs, Al, 16);

    /* Compute diff values. */
    vint16m1_t diffs = __riscv_vnot_v_i16m1_mu(mask, coefs, coefs, 16);

    /* Store transformed coefficients and diff values. */
    __riscv_vse16_v_u16m1(values_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(coefs), 16);
    __riscv_vse16_v_u16m1(diff_values_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(diffs), 16);
#endif

    values_ptr += 16;
    diff_values_ptr += 16;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 0) {
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint32m4_t off = __riscv_vle32_v_u32m4(
      (uint32_t *)(jpeg_natural_order_start), remaining_coefs);
    off = __riscv_vsll_vx_u32m4(off, 1, remaining_coefs);
    vint16m2_t zero = __riscv_vmv_v_x_i16m2(0, 16);
    vint16m2_t coefs = __riscv_vluxei32_v_i16m2_tu(zero, block, off,
                                                   remaining_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool8_t mask = __riscv_vmslt_vx_i16m2_b8(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m2_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m2(coefs, Al, 16);

    /* Compute diff values. */
    vint16m2_t diffs = __riscv_vnot_v_i16m2_mu(mask, coefs, coefs, 16);

    /* Store transformed coefficients and diff values. */
    __riscv_vse16_v_u16m2(values_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(coefs), 16);
    __riscv_vse16_v_u16m2(diff_values_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(diffs), 16);
#else
    vuint32m2_t off = __riscv_vle32_v_u32m2(
      (uint32_t *)(jpeg_natural_order_start), remaining_coefs);
    off = __riscv_vsll_vx_u32m2(off, 1, remaining_coefs);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, 16);
    vint16m1_t coefs = __riscv_vluxei32_v_i16m1_tu(zero, block, off,
                                                   remaining_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m1_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m1(coefs, Al, 16);

    /* Compute diff values. */
    vint16m1_t diffs = __riscv_vnot_v_i16m1_mu(mask, coefs, coefs, 16);

    /* Store transformed coefficients and diff values. */
    __riscv_vse16_v_u16m1(values_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(coefs), 16);
    __riscv_vse16_v_u16m1(diff_values_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(diffs), 16);
#endif

    values_ptr += 16;
    diff_values_ptr += 16;
    rows_to_zero -= 2;
  }

  /* Zero remaining memory in the values and diff_values blocks. */
  if (rows_to_zero) {
    rows_to_zero *= 8;
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint16m8_t zero = __riscv_vmv_v_x_u16m8(0, rows_to_zero);
    __riscv_vse16_v_u16m8(values_ptr, zero, rows_to_zero);
    __riscv_vse16_v_u16m8(diff_values_ptr, zero, rows_to_zero);
#else
    vuint16m4_t zero = __riscv_vmv_v_x_u16m4(0, rows_to_zero);
    __riscv_vse16_v_u16m4(values_ptr, zero, rows_to_zero);
    __riscv_vse16_v_u16m4(diff_values_ptr, zero, rows_to_zero);
#endif
  }

  /* Construct zerobits bitmap.  A set bit means that the corresponding
   * coefficient != 0.
   */
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
  vbool2_t bit_mask = __riscv_vmsne_vx_u16m8_b2(__riscv_vle16_v_u16m8(values,
    DCTSIZE2), 0, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b2_u64m1(bit_mask));
#else
  vbool4_t bit_mask = __riscv_vmsne_vx_u16m4_b4(__riscv_vle16_v_u16m4(values,
    DCTSIZE2), 0, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b4_u64m1(bit_mask));
#endif

  /* Store zerobits bitmap. */
  *zerobits = bitmap;
#endif
}


/* Data preparation for encode_mcu_AC_refine().
 *
 * The equivalent scalar C function (encode_mcu_AC_refine_prepare()) can be
 * found in jcphuff.c.
 */

HIDDEN int
HUFFMAN_ENCODER_MCU_REFINE_RVV(const JCOEF *block,
                               const int *jpeg_natural_order_start, int Sl,
                               int Al, UJCOEF *absvalues, size_t *bits)
{
#ifdef __riscv_zbb
  /* Temporary storage buffers for data used to compute the signbits bitmap
   */
  uint8_t coef_sign_bits[DCTSIZE2 / CHAR_BIT];

  UJCOEF *absvalues_ptr = absvalues;
  uint8_t *coef_sign_bits_ptr = coef_sign_bits;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint32m4_t off = __riscv_vle32_v_u32m4(
      (uint32_t *)(jpeg_natural_order_start), 16);
    off = __riscv_vsll_vx_u32m4(off, 1, 16);
    vint16m2_t coefs = __riscv_vluxei32_v_i16m2(block, off, 16);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool8_t mask = __riscv_vmslt_vx_i16m2_b8(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m2_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m2(coefs, Al, 16);

    /* Store transformed coefficients and signbits values. */
    __riscv_vse16_v_u16m2(absvalues_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(coefs), 16);
    __riscv_vsm_v_b8(coef_sign_bits_ptr, mask, 16);
#else
    vuint32m2_t off = __riscv_vle32_v_u32m2(
      (uint32_t *)(jpeg_natural_order_start), 16);
    off = __riscv_vsll_vx_u32m2(off, 1, 16);
    vint16m1_t coefs = __riscv_vluxei32_v_i16m1(block, off, 16);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m1_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m1(coefs, Al, 16);

    /* Store transformed coefficients and signbits values. */
    __riscv_vse16_v_u16m1(absvalues_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(coefs), 16);
    __riscv_vsm_v_b16(coef_sign_bits_ptr, mask, 16);
#endif

    absvalues_ptr += 16;
    coef_sign_bits_ptr += (16 / CHAR_BIT);
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 0) {
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint32m4_t off = __riscv_vle32_v_u32m4(
      (uint32_t *)(jpeg_natural_order_start), remaining_coefs);
    off = __riscv_vsll_vx_u32m4(off, 1, remaining_coefs);
    vint16m2_t zero = __riscv_vmv_v_x_i16m2(0, 16);
    vint16m2_t coefs = __riscv_vluxei32_v_i16m2_tu(zero, block, off,
                                                   remaining_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool8_t mask = __riscv_vmslt_vx_i16m2_b8(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m2_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m2(coefs, Al, 16);

    /* Store transformed coefficients and signbits values. */
    __riscv_vse16_v_u16m2(absvalues_ptr,
      __riscv_vreinterpret_v_i16m2_u16m2(coefs), 16);
    __riscv_vsm_v_b8(coef_sign_bits_ptr, mask, 16);
#else
    vuint32m2_t off = __riscv_vle32_v_u32m2(
      (uint32_t *)(jpeg_natural_order_start), remaining_coefs);
    off = __riscv_vsll_vx_u32m2(off, 1, remaining_coefs);
    vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, 16);
    vint16m1_t coefs = __riscv_vluxei32_v_i16m1_tu(zero, block, off,
                                                   remaining_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    vbool16_t mask = __riscv_vmslt_vx_i16m1_b16(coefs, 0, 16);
    coefs = __riscv_vneg_v_i16m1_mu(mask, coefs, coefs, 16);
    coefs = __riscv_vsra_vx_i16m1(coefs, Al, 16);

    /* Store transformed coefficients and signbits values. */
    __riscv_vse16_v_u16m1(absvalues_ptr,
      __riscv_vreinterpret_v_i16m1_u16m1(coefs), 16);
    __riscv_vsm_v_b16(coef_sign_bits_ptr, mask, 16);
#endif

    absvalues_ptr += 16;
    coef_sign_bits_ptr += (16 / CHAR_BIT);
    rows_to_zero -= 2;
  }

  /* Zero remaining memory in blocks. */
  if (rows_to_zero) {
    rows_to_zero *= 8;
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
    vuint16m8_t zero = __riscv_vmv_v_x_u16m8(0, rows_to_zero);
    __riscv_vse16_v_u16m8(absvalues_ptr, zero, rows_to_zero);
    __riscv_vsm_v_b4(coef_sign_bits_ptr, __riscv_vmclr_m_b4(rows_to_zero),
      rows_to_zero);
#else
    vuint16m4_t zero = __riscv_vmv_v_x_u16m4(0, rows_to_zero);
    __riscv_vse16_v_u16m4(absvalues_ptr, zero, rows_to_zero);
    __riscv_vsm_v_b8(coef_sign_bits_ptr, __riscv_vmclr_m_b8(rows_to_zero),
      rows_to_zero);
#endif
  }

  /* Construct zerobits bitmap. */
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
  vuint16m8_t abs_vals = __riscv_vle16_v_u16m8(absvalues, DCTSIZE2);
  vbool2_t bit_mask = __riscv_vmsne_vx_u16m8_b2(abs_vals, 0, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b2_u64m1(bit_mask));
#else
  vuint16m4_t abs_vals = __riscv_vle16_v_u16m4(absvalues, DCTSIZE2);
  vbool4_t bit_mask = __riscv_vmsne_vx_u16m4_b4(abs_vals, 0, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b4_u64m1(bit_mask));
#endif

  /* Store zerobits bitmap. */
  bits[0] = bitmap;

  /* Construct signbits bitmap. */
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = __riscv_vmv_x_s_u64m1_u64(__riscv_vreinterpret_v_b4_u64m1(
              __riscv_vlm_v_b4(coef_sign_bits, DCTSIZE2)));
#else
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = __riscv_vmv_x_s_u64m1_u64(__riscv_vreinterpret_v_b8_u64m1(
              __riscv_vlm_v_b8(coef_sign_bits, DCTSIZE2)));
#endif

  /* Store zerobits bitmap. */
  bits[1] = bitmap;

  /* Construct bitmap to find EOB position (the index of the last coefficient
   * equal to 1.)
   */
#ifndef USE_HUFFMAN_ENCODER_MCU_RVV_256
  bit_mask = __riscv_vmseq_vx_u16m8_b2(abs_vals, 1, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b2_u64m1(bit_mask));
#else
  bit_mask = __riscv_vmseq_vx_u16m4_b4(abs_vals, 1, DCTSIZE2);
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = __riscv_vmv_x_s_u64m1_u64(
    __riscv_vreinterpret_v_b4_u64m1(bit_mask));
#endif

  /* Return EOB position. */
  if (bitmap == 0) {
    /* EOB position is defined to be 0 if all coefficients != 1. */
    return 0;
  } else {
    return 63 - BUILTIN_CLZL(bitmap);
  }
#endif
}

#undef USE_HUFFMAN_ENCODER_MCU_RVV_256
#undef HUFFMAN_ENCODER_MCU_FIRST_RVV
#undef HUFFMAN_ENCODER_MCU_REFINE_RVV
