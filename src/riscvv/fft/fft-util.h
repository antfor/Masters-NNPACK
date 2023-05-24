#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <nnpack/fft-constants.h>

#include<stdlib.h>

// duplicate quad

/// Takes an array of four values and creates a vector repeating those
static inline __epi_2xi32 dupq_i(uint32_t *quad, long gvl)
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

static inline __epi_2xf32 dupq_f(float *quad, long gvl)
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
    return __builtin_epi_vload_indexed_2xf32(quad, indices, gvl);
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

static inline __epi_2xi32 indexA_address(__uint32_t *ind, __uint32_t N, __uint32_t jump, long gvl)
{
    __epi_2xi32 indA = indexA(ind, N, jump, gvl);
    return __builtin_epi_vmul_2xi32(indA, __builtin_epi_vmv_v_x_2xi32(4, gvl), gvl);
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

static inline __epi_2xi32 indexN_address(long gvl, __uint32_t start, __uint32_t stride, __uint32_t jump, __uint32_t N){

    return indexN_byte(gvl, start, stride, jump, N, 4);   
}


static inline __epi_2xi32 indexN(long gvl, __uint32_t start, __uint32_t stride, __uint32_t jump, __uint32_t N){

    return indexN_byte(gvl, start, stride, jump, N, 1);   
}

static inline __epi_2xi32 rvindex_adress(int start,int stride,long gvl){

    __epi_2xi32 ind_n = __builtin_epi_vid_2xi32(gvl);
    ind_n = __builtin_epi_vmul_2xi32(ind_n, __builtin_epi_vmv_v_x_2xi32(stride * 4, gvl), gvl);
   
    if(start != 0)
        ind_n =  __builtin_epi_vadd_2xi32(ind_n, __builtin_epi_vmv_v_x_2xi32(start * 4, gvl), gvl);

    return ind_n;
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

static inline __epi_2xf32 mul_twiddle_r(__epi_2xf32 br, __epi_2xf32 bi, const __epi_2xf32 tr, const __epi_2xf32 ti, long gvl)
{
    return __builtin_epi_vfsub_2xf32(__builtin_epi_vfmul_2xf32(tr, br, gvl), __builtin_epi_vfmul_2xf32(ti, bi, gvl), gvl);
}

static inline __epi_2xf32 mul_twiddle_i(__epi_2xf32 br, __epi_2xf32 bi, const __epi_2xf32 tr, const __epi_2xf32 ti, long gvl)
{
    return __builtin_epi_vfadd_2xf32(__builtin_epi_vfmul_2xf32(tr, bi, gvl), __builtin_epi_vfmul_2xf32(ti, br, gvl), gvl);
}

static inline __epi_2xi1 get_merge(long gvl){
  
    return __builtin_epi_vmslt_2xi32(__builtin_epi_vid_2xi32(gvl), __builtin_epi_vmv_v_x_2xi32(gvl/2, gvl), gvl);

}

static inline __epi_2xf32 shuffle(__epi_2xf32 a, __epi_2xf32 b, __epi_2xi32 ind, const __epi_2xi32 ind_zip, __epi_2xi1 merge, int half_VL, long gvl)
{
    a = __builtin_epi_vrgather_2xf32(a, ind, gvl);
 
    b = __builtin_epi_vrgather_2xf32(b, ind, gvl);
    b = __builtin_epi_vslideup_2xf32(b, half_VL, gvl);

    //merge =1111...0000...
    __epi_2xf32 ab = __builtin_epi_vfmerge_2xf32(b, a, merge, gvl); 

    return __builtin_epi_vrgather_2xf32(ab, ind_zip, gvl);
}

//--zip-----------------------------------------------------------------


static inline __epi_2xi32 zip_concat_4(long gvl)
{
    return indexA((uint32_t []){0, gvl/2+0}, 2, 1, gvl);
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

static inline __epi_2xi32 get_ind_even(long gvl)
{
    return __builtin_epi_vmul_2xi32(__builtin_epi_vid_2xi32(gvl),__builtin_epi_vmv_v_x_2xi32(2, gvl), gvl);
}

static inline __epi_2xi32 get_ind_odd(long gvl)
{
    __epi_2xi32 ind_np = __builtin_epi_vadd_2xi32(__builtin_epi_vid_2xi32(gvl),__builtin_epi_vmv_v_x_2xi32(1, gvl), gvl);
    ind_np = __builtin_epi_vmul_2xi32(ind_np ,__builtin_epi_vmv_v_x_2xi32(2, gvl), gvl);
    return __builtin_epi_vsub_2xi32(ind_np, __builtin_epi_vmv_v_x_2xi32(1, gvl), gvl);
}

static inline __epi_2xi32 get_ind_low_BLOCK(int BLOCK_SIZE, long gvl)
{
    return indexN(gvl, 0, 1, BLOCK_SIZE/2, BLOCK_SIZE/4);
}

static inline __epi_2xi32 get_ind_high_BLOCK(__epi_2xi32 ind_low, int BLOCK_SIZE, long gvl)
{
    return __builtin_epi_vadd_2xi32(ind_low, __builtin_epi_vmv_v_x_2xi32(BLOCK_SIZE/4, gvl), gvl);
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

//--real-----------------------------------------------------------------

static inline __epi_2xi1 get_pg_mask(int pg, long gvl){
    return __builtin_epi_vmslt_2xi32(__builtin_epi_vid_2xi32(gvl), __builtin_epi_vmv_v_x_2xi32(pg, gvl), gvl);
}

static inline __epi_2xi32 ctr_get_indr(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){
    const int offset = 2; //skip w0
    const int HL = column_count * BLOCK_SIZE;

	switch(BLOCK_SIZE){
		case 4:
			return indexA_address((uint32_t []){2,0+HL}, BLOCK_SIZE/2, BLOCK_SIZE, gvl);
			break;
		case 8:
			return indexA_address((uint32_t []){2,4,6,0+HL}, 8/2, 8, gvl);
			break;
	}

    fprintf(stderr,"Error: not a valid BLOCK_SIZE");
    return indexN_address(gvl, 1, 0, 0, 1);
}

static inline __epi_2xi32 ctr_get_indN_r(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){
    const int offset = 2; //skip w0
    const int HL = column_count * BLOCK_SIZE;

	switch(BLOCK_SIZE){
		case 4:
			return indexA_address((uint32_t []){2+HL,0+HL}, BLOCK_SIZE/2, BLOCK_SIZE, gvl);
			break;
		case 8:
			return indexA_address((uint32_t []){6+HL,4+HL,2+HL,0+HL}, 8/2, 8, gvl);
			break;
	}

    fprintf(stderr,"Error: not a valid BLOCK_SIZE");
    return indexN_address(gvl, 1, 0, 0, 1);
}


static inline __epi_2xi32 ctr_get_ind_store_top(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

    const int offset = 2; //skip w0
    const int HL = column_count * BLOCK_SIZE;
    return indexN_address(gvl, offset * N, N * 2, 1, BLOCK_SIZE/2);
}


static inline __epi_2xi32 ctr_get_ind_store_bot(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

	switch(BLOCK_SIZE){
		case 4:
			return indexA_address((uint32_t []){2*N,0*N}, BLOCK_SIZE/2, 1, gvl);
			break;
		case 8:
			return indexA_address((uint32_t []){6*16, 4*16, 2*16, 0*16}, 8/2, 1, gvl);
			break;
	}

    fprintf(stderr,"Error: not a valid BLOCK_SIZE");
    return indexN_address(gvl, 1, 0, 0, 1);
}

//-----


static inline __epi_2xi32 rtc_get_indr(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

    return  indexN_address(gvl, N, N, 1, BLOCK_SIZE/2);
}

static inline __epi_2xi32 rtc_get_indN_r(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

    return  indexN_address(gvl, N*(BLOCK_SIZE-1), -N, 1, BLOCK_SIZE/2);
}


static inline __epi_2xi32 rtc_get_ind_store_top(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

    return indexN_address(gvl, 2, 2, N, BLOCK_SIZE/2);
}


static inline __epi_2xi32 rtc_get_ind_store_bot(
	int BLOCK_SIZE, uint32_t column_count, int N,
	long gvl){

	return indexN_address(gvl, BLOCK_SIZE - 2, -2, N, BLOCK_SIZE/2);
}

//--twiddle---------------------------------------------------------------------

static inline __epi_2xf32 get_twiddle_i_top_r(int BLOCK_SIZE, long gvl){

	switch(BLOCK_SIZE){
	case 4:
		return dupq_f((float []){-SIN_1PI_OVER_4, -SIN_2PI_OVER_4, -SIN_1PI_OVER_4, -SIN_2PI_OVER_4}, gvl);
	
	case 8:
		return dupq_f((float []){-SIN_1PI_OVER_8, -SIN_2PI_OVER_8, -SIN_3PI_OVER_8, -SIN_4PI_OVER_8}, gvl);
	}

    fprintf(stderr,"Error: not a valid BLOCK_SIZE");
    return dupq_f((float []){-1,-1,-1,-1},gvl);

}

static inline __epi_2xf32 get_twiddle_i_top_i(int BLOCK_SIZE, long gvl){

	switch(BLOCK_SIZE){
	case 4:
		return dupq_f((float []){COS_1PI_OVER_4, COS_2PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4}, gvl);
	
	case 8:
		return dupq_f((float []){COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8}, gvl);
	}

    fprintf(stderr,"Error: not a valid BLOCK_SIZE");
    return dupq_f((float []){-1,-1,-1,-1},gvl);

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
