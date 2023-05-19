#pragma once

#include <stddef.h>
#include <stdint.h>
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
//todo jump not mad to byte?
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


static inline __epi_2xi32 indexN_byte(long gvl, __uint32_t start, __uint32_t stride, __uint32_t jump, __uint32_t N, int byte_size){
    
    start = start * byte_size;
    stride = stride * byte_size;
    jump = jump * byte_size;

    __epi_2xi32 ind_n = __builtin_epi_vid_2xi32(gvl);
    ind_n = __builtin_epi_vmul_2xi32(ind_n, __builtin_epi_vmv_v_x_2xi32(stride, gvl), gvl);
    __epi_2xi32 ind_div = __builtin_epi_vdiv_2xi32(ind_n, __builtin_epi_vmv_v_x_2xi32(N * stride, gvl), gvl);
    __epi_2xi32 ind_mul = __builtin_epi_vmul_2xi32(ind_div, __builtin_epi_vmv_v_x_2xi32(jump - (N * stride), gvl), gvl);
   
    return __builtin_epi_vadd_2xi32(__builtin_epi_vadd_2xi32(ind_mul,ind_n,gvl), __builtin_epi_vmv_v_x_2xi32(start, gvl) ,gvl);
}

static inline __epi_2xi32 indexN(long gvl, __uint32_t start, __uint32_t stride, __uint32_t jump, __uint32_t N){

    return indexN_byte(gvl, start, stride, jump, N, 4);   
}

//--fft-----------------------------------------------------------------

// Butterfly
static inline __epi_2xf32 butterfly_add(__epi_2xf32 a, __epi_2xf32 b, long gvl)
{
    return __builtin_epi_vfadd_2xf32(a, b, gvl);
}

static inline __epi_2xf32 butterfly_sub(__epi_2xf32 a, __epi_2xf32 b, long gvl)
{
    return __builtin_epi_vfsub_2xf32(a, b, gvl);
}

static inline __epi_2xf32 mulc_twiddle_r(__epi_2xf32 br, __epi_2xf32 bi, const __epi_2xf32 tr, const __epi_2xf32 ti, long gvl)
{
    return __builtin_epi_vfadd_2xf32(__builtin_epi_vfmul_2xf32(tr, br, gvl), __builtin_epi_vfmul_2xf32(ti, bi, gvl), gvl);
}

static inline __epi_2xf32 mulc_twiddle_i(__epi_2xf32 br, __epi_2xf32 bi, const __epi_2xf32 tr, const __epi_2xf32 ti, long gvl)
{
    return __builtin_epi_vfsub_2xf32(__builtin_epi_vfmul_2xf32(tr, bi, gvl), __builtin_epi_vfmul_2xf32(ti, br, gvl), gvl);
}

static inline __epi_2xi1 get_merge(long gvl){
  
    return __builtin_epi_vmslt_2xi32(__builtin_epi_vid_2xi32(gvl), __builtin_epi_vmv_v_x_2xi32(gvl/2, gvl), gvl);

}

static inline __epi_2xf32 shuffle(__epi_2xf32 a, __epi_2xf32 b, __epi_2xi32 ind, const __epi_2xi32 ind_zip, __epi_2xi1 merge, int half_VL, long gvl)
{
    a = __builtin_epi_vrgather_2xf32(a, ind, gvl);
    b = __builtin_epi_vrgather_2xf32(b, ind, gvl);
    b = __builtin_epi_vslideup_2xf32(b, half_VL, gvl);

    //merge =0000...1111...
    __epi_2xf32 ab = __builtin_epi_vfmerge_2xf32(a, b, merge, gvl);
    return __builtin_epi_vrgather_2xf32(a, ind_zip, gvl);
}

//--zip-----------------------------------------------------------------


static inline __epi_2xi32 zip_concat_4(long gvl)
{
    return indexA((uint32_t []){0, gvl+0}, 2, 1, gvl);
}

static inline __epi_2xi32 zip_concat_8(long gvl)
{
    return indexA((uint32_t []){0, 1, gvl/2+0, gvl/2+1}, 4, 2, gvl);
}

static inline __epi_2xi32 zip_interleave_8(long gvl)
{
    return indexA((uint32_t []){0, gvl/2+0, 1, gvl/2+1}, 4, 2, gvl);
}

static inline __epi_2xi32 zip_concat_16(long gvl)
{
    return indexA((uint32_t []){0, 1, 2, 3, gvl/2+0, gvl/2+1, gvl/2+2, gvl/2+3}, 8, 4, gvl);
}

//--ind-----------------------------------------------------------------

static inline __epi_2xi32 ind_even(long gvl)
{
    return __builtin_epi_vmul_2xi32(__builtin_epi_vid_2xi32(gvl),__builtin_epi_vmv_v_x_2xi32(2, gvl), gvl);
}

static inline __epi_2xi32 ind_odd(long gvl)
{
    __epi_2xi32 ind_np = __builtin_epi_vadd_2xi32(__builtin_epi_vid_2xi32(gvl),__builtin_epi_vmv_v_x_2xi32(1, gvl), gvl);
    ind_np = __builtin_epi_vmul_2xi32(ind_np ,__builtin_epi_vmv_v_x_2xi32(2, gvl), gvl);
    return __builtin_epi_vsub_2xi32(ind_np, __builtin_epi_vmv_v_x_2xi32(1, gvl), gvl);
}

static inline __epi_2xi32 ind_low_BLOCK(int BLOCK_SIZE, long gvl)
{
    return indexN(gvl, 0, 1, BLOCK_SIZE, BLOCK_SIZE/2);
}

static inline __epi_2xi32 ind_high_BLOCK(__epi_2xi32 ind_low, int BLOCK_SIZE, long gvl)
{
    return __builtin_epi_vadd_2xi32(ind_low, __builtin_epi_vmv_v_x_2xi32(BLOCK_SIZE/2, gvl), gvl);
}

//--gemm-----------------------------------------------------------------

static inline __epi_2xf32 add_halfs(__epi_2xf32 a, int half_point, long gvl){
    return __builtin_epi_vfadd_2xf32(a, __builtin_epi_vslidedown_2xf32(a, half_point, gvl) , gvl);
}

static inline __epi_2xf32 sumSplit(__epi_2xf32 rvAcc, int split, int simd_width, long gvl){


    if(split == 2){
        rvAcc = add_halfs(rvAcc, simd_width, gvl);
    }else if(split % 2 == 0 && split > 2){
        return sumSplit(add_halfs(rvAcc, split/2 * simd_width, gvl), split/2, simd_width, gvl);
    }else{
        fprintf(stderr,"Error: split must be a power of 2");
    }

    return rvAcc;

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
