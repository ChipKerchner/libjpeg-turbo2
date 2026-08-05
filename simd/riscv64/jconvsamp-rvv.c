/*
 * Integer Sample Conversion (64-bit RVV 1.0)
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

#ifndef CONVSAMP_RVV
#ifdef USE_CONVSAMP_RVV_256
#define CONVSAMP_RVV         jsimd_convsamp_rvv_vlen256
#else
#define CONVSAMP_RVV         jsimd_convsamp_rvv
#endif
#endif

#ifdef USE_CONVSAMP_RVV_256
#include "../jsimdint.h"
#include <riscv_vector.h>
#endif


HIDDEN void
CONVSAMP_RVV(JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace)
{
  /* The minimum register width (VLEN) for standard CPUs in RVV 1.0 is
   * 128 bits.  Thus, this should always be 8, meaning that only one pass is
   * required.
   */
#ifdef USE_CONVSAMP_RVV_256
  size_t vl = __riscv_vsetvl_e16mf2(DCTSIZE);

  vuint8mf4_t in0, in1, in2, in3, in4, in5, in6, in7;
  vint16mf2_t row0, row1, row2, row3, row4, row5, row6, row7;

  in0 = __riscv_vle8_v_u8mf4(sample_data[0] + start_col, vl);
  in1 = __riscv_vle8_v_u8mf4(sample_data[1] + start_col, vl);
  in2 = __riscv_vle8_v_u8mf4(sample_data[2] + start_col, vl);
  in3 = __riscv_vle8_v_u8mf4(sample_data[3] + start_col, vl);
  in4 = __riscv_vle8_v_u8mf4(sample_data[4] + start_col, vl);
  in5 = __riscv_vle8_v_u8mf4(sample_data[5] + start_col, vl);
  in6 = __riscv_vle8_v_u8mf4(sample_data[6] + start_col, vl);
  in7 = __riscv_vle8_v_u8mf4(sample_data[7] + start_col, vl);

  row0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in0, CENTERJSAMPLE, vl));
  row1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in1, CENTERJSAMPLE, vl));
  row2 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in2, CENTERJSAMPLE, vl));
  row3 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in3, CENTERJSAMPLE, vl));
  row4 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in4, CENTERJSAMPLE, vl));
  row5 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in5, CENTERJSAMPLE, vl));
  row6 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in6, CENTERJSAMPLE, vl));
  row7 = __riscv_vreinterpret_v_u16mf2_i16mf2(
           __riscv_vwsubu_vx_u16mf2(in7, CENTERJSAMPLE, vl));

  __riscv_vse16_v_i16mf2(workspace + 0 * DCTSIZE, row0, vl);
  __riscv_vse16_v_i16mf2(workspace + 1 * DCTSIZE, row1, vl);
  __riscv_vse16_v_i16mf2(workspace + 2 * DCTSIZE, row2, vl);
  __riscv_vse16_v_i16mf2(workspace + 3 * DCTSIZE, row3, vl);
  __riscv_vse16_v_i16mf2(workspace + 4 * DCTSIZE, row4, vl);
  __riscv_vse16_v_i16mf2(workspace + 5 * DCTSIZE, row5, vl);
  __riscv_vse16_v_i16mf2(workspace + 6 * DCTSIZE, row6, vl);
  __riscv_vse16_v_i16mf2(workspace + 7 * DCTSIZE, row7, vl);
#else
  size_t vl = __riscv_vsetvl_e16m1(DCTSIZE);

  vuint8mf2_t in0, in1, in2, in3, in4, in5, in6, in7;
  vint16m1_t row0, row1, row2, row3, row4, row5, row6, row7;

  in0 = __riscv_vle8_v_u8mf2(sample_data[0] + start_col, vl);
  in1 = __riscv_vle8_v_u8mf2(sample_data[1] + start_col, vl);
  in2 = __riscv_vle8_v_u8mf2(sample_data[2] + start_col, vl);
  in3 = __riscv_vle8_v_u8mf2(sample_data[3] + start_col, vl);
  in4 = __riscv_vle8_v_u8mf2(sample_data[4] + start_col, vl);
  in5 = __riscv_vle8_v_u8mf2(sample_data[5] + start_col, vl);
  in6 = __riscv_vle8_v_u8mf2(sample_data[6] + start_col, vl);
  in7 = __riscv_vle8_v_u8mf2(sample_data[7] + start_col, vl);

  row0 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in0, CENTERJSAMPLE, vl));
  row1 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in1, CENTERJSAMPLE, vl));
  row2 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in2, CENTERJSAMPLE, vl));
  row3 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in3, CENTERJSAMPLE, vl));
  row4 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in4, CENTERJSAMPLE, vl));
  row5 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in5, CENTERJSAMPLE, vl));
  row6 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in6, CENTERJSAMPLE, vl));
  row7 = __riscv_vreinterpret_v_u16m1_i16m1(
           __riscv_vwsubu_vx_u16m1(in7, CENTERJSAMPLE, vl));

  __riscv_vse16_v_i16m1(workspace + 0 * DCTSIZE, row0, vl);
  __riscv_vse16_v_i16m1(workspace + 1 * DCTSIZE, row1, vl);
  __riscv_vse16_v_i16m1(workspace + 2 * DCTSIZE, row2, vl);
  __riscv_vse16_v_i16m1(workspace + 3 * DCTSIZE, row3, vl);
  __riscv_vse16_v_i16m1(workspace + 4 * DCTSIZE, row4, vl);
  __riscv_vse16_v_i16m1(workspace + 5 * DCTSIZE, row5, vl);
  __riscv_vse16_v_i16m1(workspace + 6 * DCTSIZE, row6, vl);
  __riscv_vse16_v_i16m1(workspace + 7 * DCTSIZE, row7, vl);
#endif
}

#undef USE_CONVSAMP_RVV_256
#undef CONVSAMP_RVV
