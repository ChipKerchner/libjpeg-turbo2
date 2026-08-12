/*
 * Quantization (64-bit RVV 1.0)
 *
 * Copyright (C) 2022-2023, Institute of Software, Chinese Academy of Sciences.
 *                          Author:  Zhiyuan Tan
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2026, D. R. Commander.
 * Copyright (C) 2026, Chip Kerchner
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

#include "../jsimdint.h"
#include <riscv_vector.h>


HIDDEN void
jsimd_quantize_rvv(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  int coeffs_remaining = DCTSIZE2;

  vint16m4_t in, shift, out;
  vuint16m4_t recip, corr, temp;
  vuint32m8_t product;
  vbool4_t mask;

  while (coeffs_remaining > 0) {
    /* vl = the number of 16-bit elements that can be stored in a 4-deep RVV
     * register group, up to a maximum of coeffs_remaining.  For example, this
     * will be 64 if the register width (VLEN) is 256, meaning that all 64
     * coefficients in the block can be processed in one pass.
     */
    size_t vl = __riscv_vsetvl_e16m4(coeffs_remaining);

    in = __riscv_vle16_v_i16m4(workspace, vl);
    recip = __riscv_vle16_v_u16m4((UDCTELEM *)divisors, vl);
    corr = __riscv_vle16_v_u16m4((UDCTELEM *)divisors + DCTSIZE2, vl);
    shift = __riscv_vle16_v_i16m4(divisors + 3 * DCTSIZE2, vl);

    /* mask[i] = in[i] < 0 ? 1 : 0 */
    mask = __riscv_vmslt_vx_i16m4_b4(in, 0, vl);
    /* Compute absolute value. */
    in = __riscv_vneg_v_i16m4_mu(mask, in, in, vl);
    temp = __riscv_vreinterpret_v_i16m4_u16m4(in);

    temp = __riscv_vadd_vv_u16m4(temp, corr, vl);
    temp = __riscv_vmulhu_vv_u16m4(temp, recip, vl);
    temp = __riscv_vsrl_vv_u16m4(temp,
             __riscv_vreinterpret_v_i16m4_u16m4(shift), vl);

    out = __riscv_vreinterpret_v_u16m4_i16m4(temp);
    /* Restore sign to original product. */
    out = __riscv_vneg_v_i16m4_mu(mask, out, out, vl);
    __riscv_vse16_v_i16m4(coef_block, out, vl);

    workspace += vl;
    divisors += vl;
    coef_block += vl;
    coeffs_remaining -= vl;
  }
}
