#pragma once

#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#include<stdlib.h>

// duplicate quad

/// Takes an array of four values and creates a vector repeating those
static inline __epi_2xi32 dupq(__uint32_t *quad, long gvl)
{

    // ind_a = 0, 1, 2, 3, 4, 5, 6, 7, 8
    __epi_2xi32 increasing = __builtin_epi_vid_2xi32(gvl);

    // 4, 4, 4, 4...
    __epi_2xi32 four = __builtin_epi_vmv_v_x_2xi32(4, gvl);

    // quads = 0, 0, 0, 0, 1, 1, 1, 1...
    __epi_2xi32 quads = __builtin_epi_vdiv_2xi32(increasing, four, gvl);
    // quads = 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8...
    quads = __builtin_epi_vmul_2xi32(quads, four, gvl);

    // increasing - 4 * quads = 0, 1, 2, 3, 0, 1, 2, 3, 
    __epi_2xi32 indices = __builtin_epi_vsub_2xi32(increasing, quads, gvl);
    // multiply by four for byte adressing in RVV
    indices = __builtin_epi_vmul_2xi32(indices, four, gvl);

    // Load from indices to get vector of 4 repeating values
    return __builtin_epi_vload_indexed_unsigned_2xi32(quad, indices, gvl);
}

//--index--------------------------------------------------------------

static inline __epi_2xi32 indexA(__uint32_t *ind, __uint32_t N, __uint32_t jump, long gvl)
{
    __epi_2xi32 ind_n = __builtin_epi_vid_2xi32(gvl);
    __epi_2xi32 div = __builtin_epi_vdiv_2xi32(ind_n, __builtin_epi_vmv_v_x_2xi32(N, gvl), gvl);
    __epi_2xi32 jumps = __builtin_epi_vmul_2xi32(div, __builtin_epi_vmv_v_x_2xi32(jump, gvl), gvl);
    __epi_2xi32 repeat = __builtin_epi_vadd_2xi32(ind_n, __builtin_epi_vmul_2xi32(div ,__builtin_epi_vmv_v_x_2xi32(- N, gvl), gvl), gvl);
    repeat = __builtin_epi_vmul_2xi32(repeat, __builtin_epi_vmv_v_x_2xi32(sizeof(__uint32_t), gvl), gvl);
    __epi_2xi32 load = __builtin_epi_vload_indexed_unsigned_2xi32(ind, repeat, gvl);
    return __builtin_epi_vadd_2xi32(load, jumps, gvl);
}


// Butterfly

static inline void butterfly(__epi_2xf32 *a, __epi_2xf32 *b, __epi_2xf32 *new_a, __epi_2xf32 *new_b, long gvl)
{
    *new_a = __builtin_epi_vfadd_2xf32(*a, *b, gvl);
    *new_b = __builtin_epi_vfadd_2xf32(*a, *b, gvl);
}

static inline void suffle(__epi_2xf32 *a, __epi_2xf32 *b, const __epi_2xi32 *ind_a, const __epi_2xi32 *ind_b, const __epi_2xi32 *ind_zip, __epi_2xf32 *new_a, __epi_2xf32 *new_b)
{

    *new_a = svtbl(svzip1(svtbl(*a, *ind_a), svtbl(*b, *ind_a)), *ind_zip);
    *new_b = svtbl(svzip1(svtbl(*a, *ind_b), svtbl(*b, *ind_b)), *ind_zip);
}

//----gp-utils-----------------------------------------------------------------

static inline int imin(int a, int b){
	return a < b ? a : b;
}


static inline int imax(int a, int b){
	return a > b ? a : b;
}

static inline int idiv_ceil(int a, int b){
    return (a + b - 1) / b;
} 
