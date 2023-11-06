#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>


//--4xN-------------------------------------------------------------
//todo

//--8xN-------------------------------------------------------------


static inline void fft8xNr_channel(
	float t_lo[restrict static 1],
	float t_hi[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	uint32_t column_offset, uint32_t column_count,
	float tf[restrict static 1],int channels){

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_LENGTH = column_count * BLOCK_SIZE;
	
	uint32_t numVals = svcntw() / BLOCK_SIZE;
	svbool_t pg, pg_a, pg_b;
	const svbool_t all = svptrue_b32();
	svuint32_t t_lo_offset, t_hi_offset;

	svfloat32_t a, b, new_a, new_b, new_bt;

	const svuint32_t ind_zip = zip_concat_8(all);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);


	aos8_pred_and_offset(row_offset, row_count, &pg_a, &pg_b, stride_t, &t_lo_offset, &t_hi_offset);


	const int channel_stride = column_count * row_count;
	const int tf_stride = 16 * column_count;
	const svuint32_t ind_strore = indexN(all,0, 1, tf_stride, BLOCK_SIZE);
	//jump value can be diffret for riscV?
	const svuint32_t channel_offset = repeatN(all, 0, channel_stride * 4 - 4, BLOCK_SIZE);

	t_lo_offset = svadd_m(all,t_lo_offset, channel_offset);
	t_hi_offset = svadd_m(all,t_hi_offset, channel_offset);


	for(uint32_t column = 0; column < column_count; column += 1){

		for(uint32_t channel = 0; channel < channels; channel += numVals){
	
			pg = svwhilelt_b32_s32(channel * BLOCK_SIZE , channels * BLOCK_SIZE);

			// load
			a = svld1_gather_offset(svmov_z(pg, pg_a), t_lo + column + channel * channel_stride, t_lo_offset);
			b = svld1_gather_offset(svmov_z(pg, pg_b), t_hi + column + channel * channel_stride, t_hi_offset);

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

			svst1_scatter_index(pg, tf + channel * tf_stride + column * BLOCK_SIZE + 0, ind_strore, new_a);
			svst1_scatter_index(pg, tf + channel * tf_stride + column * BLOCK_SIZE + HALF_BLOCK_LENGTH, ind_strore, new_b);
		}

	}
}