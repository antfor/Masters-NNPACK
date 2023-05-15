#pragma once

#include <nnpack/fft-constants.h>
#include <riscvv/fft/aos.h>
#include <riscvv/fft/fft-util.h>
#include <riscvv/fft/complex.h>

inline static void fft4xNr( 
	const float t_lo[restrict static 1],
	const float t_hi[restrict static 1],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f[restrict static 1],
	const uint32_t N)
{

	const uint32_t BLOCK_SIZE = 4;
	const uint32_t LENGTH = BLOCK_SIZE * N;

	const uint64_t max_32 = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);

    long gvl = __builtin_epi_vsetvl(max_32, __epi_e32, __epi_m1);

	//const __epi_2xi32 twiddle = dupq((uint32_t []){COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2}, gvl);
	const __epi_2xi32 twiddle_r = dupq((uint32_t []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
	const __epi_2xi32 twiddle_i = dupq((uint32_t []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);

	const __epi_2xi32 ind_zip   = indexA((uint32_t []){0, 2, 1, 3}, 4, 4, gvl);
	const __epi_2xi32 ind_low   = indexA((uint32_t []){0, 1}, 2, 4, gvl);
	const __epi_2xi32 ind_high  = indexA((uint32_t []){2, 3}, 2, 4, gvl);
	//const __epi_2xi32 ind_store = indexA((uint32_t []){0, 4, 8, 12}, 4, 8 * 4, gvl);
	const __epi_2xi32 ind_store_r = indexA((uint32_t []){0 * 4, 2 * 4}, 2, 8 * 4, gvl);
	const __epi_2xi32 ind_store_i = indexA((uint32_t []){1 * 4, 3 * 4}, 2, 8 * 4, gvl);

	// Offsets and masks

    const uint32_t row_end = row_start + row_count;
    const bool a_pred[4] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3};
	__epi_2xi32 t_lo_offset_r = aos4_offset_r(&a_pred, stride_t, gvl);
	__epi_2xi32 t_lo_offset_i = aos4_offset_i(&a_pred, stride_t, gvl);

    const bool b_pred[4] = {row_start <= 4 && row_end > 4, row_start <= 5 && row_end > 5, row_start <= 6 && row_end > 6, row_start <= 7 && row_end > 7};
	__epi_2xi32 t_hi_offset_r = aos4_offset_r(&b_pred, stride_t, gvl);
	__epi_2xi32 t_hi_offset_i = aos4_offset_i(&b_pred, stride_t, gvl);

    bool no_jump[8];
	jump_arr(&no_jump, &a_pred, &b_pred, row_count);

	__epi_2xi1 mask_a_r = aos4_mask_a_r(&no_jump, &a_pred, &b_pred, gvl);
	__epi_2xi1 mask_a_i = aos4_mask_a_i(&no_jump, &a_pred, &b_pred, gvl);
	__epi_2xi1 mask_b_r = aos4_mask_b_r(&no_jump, &a_pred, &b_pred, gvl);
	__epi_2xi1 mask_b_i = aos4_mask_b_i(&no_jump, &a_pred, &b_pred, gvl);

	for (uint32_t i = 0; i < LENGTH)
	{
		long gvl = __builtin_epi_vsetvl(min(max_32, LENGTH - max_32), __epi_e32, __epi_m1);

		// load
		__epi_2xi32 a_r = __builtin_epi_vload_indexed_2xi32_mask(__builtin_epi_vmv_v_x_2xi32(0), t_lo + i / BLOCK_SIZE, t_lo_offset_r, mask_a_r, gvl);
		__epi_2xi32 a_i = __builtin_epi_vload_indexed_2xi32_mask(__builtin_epi_vmv_v_x_2xi32(0), t_lo + i / BLOCK_SIZE, t_lo_offset_i, mask_a_i, gvl);
		__epi_2xi32 b_r = __builtin_epi_vload_indexed_2xi32_mask(__builtin_epi_vmv_v_x_2xi32(0), t_hi + i / BLOCK_SIZE, t_hi_offset_r, mask_b_r, gvl);
		__epi_2xi32 b_i = __builtin_epi_vload_indexed_2xi32_mask(__builtin_epi_vmv_v_x_2xi32(0), t_hi + i / BLOCK_SIZE, t_hi_offset_i, mask_b_i, gvl);

		// stage1
		__epi_2xi32 new_a_r = butterfly_add(a_r, b_r, gvl);
		__epi_2xi32 new_a_i = butterfly_add(a_i, b_i, gvl);
		__epi_2xi32 new_b_r = butterfly_sub(a_r, b_r, gvl);
		__epi_2xi32 new_b_i = butterfly_sub(a_i, b_i, gvl);

		__epi_2xf32 new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);
		__epi_2xf32 new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);

		new_a_r = shuffle(&new_a_r, &new_bt_r, &ind_low, &ind_zip, gvl);
		new_a_i = shuffle(&new_a_i, &new_bt_i, &ind_low, &ind_zip, gvl);
		new_b_r = shuffle(&new_a_r, &new_bt_r, &ind_high, &ind_zip, gvl);
		new_b_r = shuffle(&new_a_i, &new_bt_i, &ind_high, &ind_zip, gvl);

		// stage2
		__epi_2xi32 new_a_r = butterfly_add(a_r, b_r, gvl);
		__epi_2xi32 new_a_i = butterfly_add(a_i, b_i, gvl);
		__epi_2xi32 new_b_r = butterfly_sub(a_r, b_r, gvl);
		__epi_2xi32 new_b_i = butterfly_sub(a_i, b_i, gvl);

		// store

		// real and img, need to do every other2
		// 0, 1, 2, 3, 8, 9, 10, 11

		// should be 0, 2, 8, 10,
		__builtin_epi_vstore_indexed_2xf32(f + i * 2 + 0, new_a_r, ind_store_r, gvl);
		// should be 1, 3, 9, 11
		__builtin_epi_vstore_indexed_2xf32(f + i * 2 + 0, new_a_i, ind_store_i, gvl);
		// 4, 5, 6, 7, 12, 13, 14, 15
		// should be 4, 6, 12, 14,
		__builtin_epi_vstore_indexed_2xf32(f + i * 2 + 4, new_b_r, ind_store_r, gvl);
		// should be 5, 7, 13, 15
		__builtin_epi_vstore_indexed_2xf32(f + i * 2 + 4, new_b_i, ind_store_i, gvl);

		i += gvl;
	}
}

static inline void riscvv_fft8xN_real(
	const float t0[restrict static 1],
	const float t4[restrict static 1],
	size_t stride_t,
	const uint32_t row_offset, 
	const uint32_t row_count,
	const uint32_t column_offset,
	const uint32_t column_count,
	float f[restrict static 1])
{
	float w[8 * column_count];

	fft4xNr(t0, t4, stride_t, row_offset, row_count, w, column_count);

	complex_to_real_NxNc(w, f, column_offset, column_count, 8);
	fftN_to_fft2N(w,4,f,8,column_offset,column_count);
}

static inline void complex_to_real_NxNc(
	const float w[restrict static 1], 
	float f[restrict static 1],
	uint32_t column_offset, uint32_t column_count, 
	int N){

	const uint32_t BLOCK_SIZE = N/2;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;

	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);
	const svfloat32_t i = svdupq_f32(0.0f, 1.0f, 0.0f, 1.0f);

	svbool_t pg;
	svfloat32_t xr, xN_r,x, xe, xo, xot;
	svuint32_t indr, indN_r, ind_store_top, ind_store_bot;
	ctr_get_index(BLOCK_SIZE, column_count, N, &indr, &indN_r, &ind_store_top, &ind_store_bot);
	svfloat32_t twiddle_i = ctr_get_twiddle_i_top(BLOCK_SIZE);

	for(int column = 0; column < column_count; column+=numVals){

		pg = svwhilelt_b32_s32(column * BLOCK_SIZE, column_count * BLOCK_SIZE);

		//load 
		xr  =  svld1_gather_index(pg, w + column * BLOCK_SIZE , indr);
		xN_r = svld1_gather_index(pg, w + column * BLOCK_SIZE , indN_r);

		xN_r = svmul_m(pg, xN_r, to_conjugate);

		xe = svadd_m(pg, xr, xN_r);
		xe = svmul_m(pg, xe, 0.5f);

		xo = svsub_m(pg, xr, xN_r);
		xo = svmul_m(pg, xo, 0.5f);

		cmulc_twiddle(&pg, &xo, &twiddle_i, &xot);

		x = svadd_m(pg, xe, xot);

		svst1_scatter_index(pg, f + column_offset + column + 0, ind_store_top, x);

		x = svsub_m(pg, xe, xot);
		x = svmul_m(pg, x, to_conjugate);

		svst1_scatter_index(pg, f + column_offset + column + BLOCK_SIZE * N, ind_store_bot, x);
 
	}
}

inline static void fftN_to_fft2N(
	const float w[restrict static 1],
	size_t stride_w,

	float f[restrict static 1],
    size_t stride_f,
    const uint32_t column_offset,
	const uint32_t column_count)
{

    svfloat32_t a, b, new_a, new_b;
    svbool_t pg;
    uint32_t numVals = svcntw();

    const svuint32_t ind = svindex_u32(0, stride_w);

    for(int column =0; column < column_count; column += numVals ){
        pg = svwhilelt_b32_s32(column, column_count);

        //load
        a = svld1_gather_index(pg, w + column * stride_w + 0, ind);
        b = svld1_gather_index(pg, w + column * stride_w + 1, ind);
    
        //stage 1
        butterfly(&pg, &a, &b, &new_a, &new_b);

        //store
         svst1(pg, f + column_offset + column + 0, new_a);
         svst1(pg, f + column_offset + column + stride_f, new_b);
    }

s