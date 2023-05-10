#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>


//--4xN-------------------------------------------------------------

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

	svbool_t pg, pg_a, pg_b;
	svuint32_t t_lo_offset, t_hi_offset;
	svfloat32_t b, a, new_b, new_a, new_bt;

	const uint64_t numVals = svcntw();

	const svfloat32_t twiddle = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

	const svuint32_t ind_zip = index4(0, 2, 1, 3, 4);
	const svuint32_t ind_low = index2(0, 1, 4);
	const svuint32_t ind_high = index2(2, 3, 4);

	aos4_pred_and_offset(row_start, row_count, &pg_a, &pg_b, stride_t, &t_lo_offset, &t_hi_offset);

	for (uint32_t i = 0; i < LENGTH; i += numVals)
	{
		pg = svwhilelt_b32_s32(i, LENGTH);

		// load
		a = svld1_gather_offset(svmov_z(pg, pg_a), t_lo + i / BLOCK_SIZE, t_lo_offset);
		b = svld1_gather_offset(svmov_z(pg, pg_b), t_hi + i / BLOCK_SIZE, t_hi_offset);

		// stage1
		butterfly(&pg, &a, &b, &new_a, &new_b);
		cmulc_twiddle(&pg, &new_b, &twiddle, &new_bt);
		suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

		// stage2
		butterfly(&pg, &a, &b, &new_a, &new_b);

		// store
		svst1(pg, f + i + 0, new_a);
		svst1(pg, f + i + LENGTH, new_b);
	}
}

//--8xN-------------------------------------------------------------


static inline void fft8xNr(
	const float t_lo[restrict static 1],
	const float t_hi[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	uint32_t column_offset, uint32_t column_count,
	float tf[restrict static 1]){

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_LENGTH = column_count * BLOCK_SIZE;
	
	uint32_t numVals = svcntw() / BLOCK_SIZE;
	svbool_t pg, pg_a, pg_b;
	svuint32_t t_lo_offset, t_hi_offset;

	svfloat32_t a, b, new_a, new_b, new_bt;

	const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);

	aos8_pred_and_offset(row_offset, row_count, &pg_a, &pg_b, stride_t, &t_lo_offset, &t_hi_offset);

	for(uint32_t column = 0; column < column_count; column += numVals){

		pg = svwhilelt_b32_s32(column * BLOCK_SIZE, column_count * BLOCK_SIZE);

		// load
		a = svld1_gather_offset(svmov_z(pg, pg_a), t_lo + column, t_lo_offset);
		b = svld1_gather_offset(svmov_z(pg, pg_b), t_hi + column, t_hi_offset);

		// stage1
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_1, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

        // stage2
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_2, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_even, &ind_odd, &ind_zip, &a, &b);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // store
        svst1(pg, tf + column * BLOCK_SIZE + 0, new_a);
        svst1(pg, tf + column * BLOCK_SIZE + HALF_BLOCK_LENGTH, new_b);

	}
}




static inline void sve_ifft8xNr(float w[restrict static 1] ,float f[restrict static 256], size_t column_count){
	
	const svfloat32_t scaled_twiddle_1 = svzip1(svdupq_f32(0.125f * COS_0PI_OVER_4, 0.125f * COS_1PI_OVER_4, 0.125f * COS_2PI_OVER_4, 0.125f * COS_3PI_OVER_4), svdupq_f32(0.125f * SIN_0PI_OVER_4, 0.125f * SIN_1PI_OVER_4, 0.125f * SIN_2PI_OVER_4, 0.125f * SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_LENGTH = column_count * BLOCK_SIZE;

	uint32_t numVals = svcntw() / BLOCK_SIZE;
	svbool_t pg;
	const svbool_t all = svptrue_b32();

	svfloat32_t a, b, new_a, new_b, new_bt;

	const svuint32_t ind_zip_interleave = zip_interleave_8(all);
    const svuint32_t ind_zip_concat = zip_concat_8(all);
    const svuint32_t ind_low = indexN(all, 0, 1, 8, 4);
    const svuint32_t ind_high = svadd_m(all, ind_low, 4);

	const svuint32_t index = indexN(all, 0, 1, 16, 8);

	for(int column = 0; column < column_count; column+=numVals){

		pg = svwhilelt_b32_s32(column * BLOCK_SIZE, column_count * BLOCK_SIZE);

		a = svld1_gather_index(pg, w + column * BLOCK_SIZE * 2 + 0, index);
        b = svld1_gather_index(pg, w + column * BLOCK_SIZE * 2 + BLOCK_SIZE, index);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // stage2
        suffle(&pg, &new_a, &new_b, &ind_low, &ind_high, &ind_zip_interleave, &a, &b);
        cmul_twiddle(&pg, &b, &twiddle_2, &new_bt);
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // stage1
        suffle(&pg, &new_a, &new_b, &ind_low, &ind_high, &ind_zip_concat, &a, &b);
        cmul_twiddle(&pg, &b, &scaled_twiddle_1, &new_bt);
        a = svmul_m(pg, a, svdup_f32(0.125f));
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // store
        svst1_scatter_index(pg, f + column * BLOCK_SIZE * 2 + 0, index, new_a);
        svst1_scatter_index(pg, f + column * BLOCK_SIZE * 2 + BLOCK_SIZE, index, new_b);
	}

}