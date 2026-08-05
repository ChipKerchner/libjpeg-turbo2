#define CONVSAMP_256_RVV            jsimd_convsamp_rvv_vlen256

#define USE_CONVSAMP_RVV_256
#define CONVSAMP_RVV                CONVSAMP_256_RVV

#include "jconvsamp-rvv.c"

#define CONVSAMP_RVV                jsimd_convsamp_rvv

#include "jconvsamp-rvv.c"
