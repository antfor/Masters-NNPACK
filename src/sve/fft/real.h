#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>

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
	const svuint32_t ind_store = index4(0 * 4, 1 * 4, 2 * 4, 3 * 4, 8 * 4);

	aos4_pred_and_offset(row_start, row_count, &pg_a, &pg_b, stride_t, &t_lo_offset, &t_hi_offset);

	for (uint32_t i = 0; i < LENGTH; i += numVals)
	{
		pg = svwhilelt_b32_s32(i, LENGTH);

		// load
		a = svld1_gather_offset(svmov_z(pg, pg_a), t_lo + i / BLOCK_SIZE, t_lo_offset);
		b = svld1_gather_offset(svmov_z(pg, pg_b), t_hi + i / BLOCK_SIZE, t_hi_offset);

		// stage1
		butterfly(&pg, &a, &b, &new_a, &new_b);
		cmul_twiddle(&pg, &new_b, &twiddle, &new_bt);
		suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

		// stage2
		butterfly(&pg, &a, &b, &new_a, &new_b);

		// store
		svst1_scatter_offset(pg, f + i * 2 + 0, ind_store, new_a);
		svst1_scatter_offset(pg, f + i * 2 + 4, ind_store, new_b);
	}
}

static inline void stuff_real8_needs_to_do_sve(
	const float w[restrict static 1],
	uint32_t column_count,
	float f[restrict static 1],
	size_t stride_f)
{

	svbool_t pg;
	svfloat32_t w0, w1, w2, w3, w4, w5, w6, w7;
	svfloat32_t g1r, g1i, two_h1r, two_h1i, h1_plus, h1_minus;
	svfloat32_t f0, f4, f1r, f1i, f2r, f2i, f3r, f3i;

	const uint32_t BLOCK_SIZE = 8;
	const uint64_t numVals = svcntw();
	const svuint32_t ind_load = svindex_u32(0, 4 * 8);

	const svfloat32_t half = svdup_f32(0.5f);
	const svfloat32_t sqrt2_over_4 = svdup_f32(SQRT2_OVER_4);

	for (uint32_t column = 0; column < column_count; column += numVals)
	{
		pg = svwhilelt_b32_s32(column, column_count);

		w0 = svld1_gather_offset(pg, w + 0 + column * BLOCK_SIZE, ind_load);
		w1 = svld1_gather_offset(pg, w + 1 + column * BLOCK_SIZE, ind_load);
		w2 = svld1_gather_offset(pg, w + 2 + column * BLOCK_SIZE, ind_load);
		w3 = svld1_gather_offset(pg, w + 3 + column * BLOCK_SIZE, ind_load);
		w4 = svld1_gather_offset(pg, w + 4 + column * BLOCK_SIZE, ind_load);
		w5 = svld1_gather_offset(pg, w + 5 + column * BLOCK_SIZE, ind_load);
		w6 = svld1_gather_offset(pg, w + 6 + column * BLOCK_SIZE, ind_load);
		w7 = svld1_gather_offset(pg, w + 7 + column * BLOCK_SIZE, ind_load);

		g1r = svmul_m(pg, half, svadd_m(pg, w6, w2));
		g1i = svmul_m(pg, half, svsub_m(pg, w3, w7));
		two_h1r = svadd_m(pg, w3, w7);
		two_h1i = svsub_m(pg, w6, w2);

		h1_plus = svmul_m(pg, sqrt2_over_4, svadd_m(pg, two_h1i, two_h1r));
		h1_minus = svmul_m(pg, sqrt2_over_4, svsub_m(pg, two_h1i, two_h1r));

		f0 = svadd_m(pg, w0, w1);
		f4 = svsub_m(pg, w0, w1);

		f1r = svadd_m(pg, g1r, h1_plus);
		f1i = svadd_m(pg, h1_minus, g1i);

		f2r = w4;
		f2i = svmul_m(pg, w5, svdup_f32(-1.0f));

		f3r = svsub_m(pg, g1r, h1_plus);
		f3i = svsub_m(pg, h1_minus, g1i);


		svst1(pg, f + 0 * stride_f + column, f0);
		svst1(pg, f + 1 * stride_f + column, f4);
		svst1(pg, f + 2 * stride_f + column, f1r);
		svst1(pg, f + 3 * stride_f + column, f1i);
		svst1(pg, f + 4 * stride_f + column, f2r);
		svst1(pg, f + 5 * stride_f + column, f2i);
		svst1(pg, f + 6 * stride_f + column, f3r);
		svst1(pg, f + 7 * stride_f + column, f3i);
	}
}

static inline void stuff_real8_needs_to_do(
	const float w[restrict static 1],
	uint32_t column_count,
	float f[restrict static 1],
	size_t stride_f)
{

	for(uint32_t column = 0; column < column_count; column++){ // todo vectorize, move to fft4xNr?
		int offset = column *8;

		const float half = 0.5f;
		const float g1r = half * (w[2 + offset] + w[6 + offset]);
		const float g1i = half * (w[3 + offset] - w[7 + offset]);
		const float two_h1r = w[3 + offset] + w[7 + offset];
		const float two_h1i = w[6 + offset] - w[2 + offset];

		const float sqrt2_over_4 = SQRT2_OVER_4;
		const float h1_plus  = sqrt2_over_4 * (two_h1i + two_h1r);
		const float h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

		const float f0 = w[0 + offset] + w[1 + offset];
		const float f4 = w[0 + offset] - w[1 + offset];
		const float f1r = g1r + h1_plus;
		const float f1i = h1_minus + g1i;
		const float f2r =  w[4 + offset];
		const float f2i = -w[5 + offset];
		const float f3r = g1r - h1_plus;
		const float f3i = h1_minus - g1i;

		/* Store outputs */
		f[0 * stride_f] = f0;
		f[1 * stride_f] = f4;
		f[2 * stride_f] = f1r;
		f[3 * stride_f] = f1i;
		f[4 * stride_f] = f2r;
		f[5 * stride_f] = f2i;
		f[6 * stride_f] = f3r;
		f[7 * stride_f] = f3i;

		f += 1;
	}

}

static inline void sve_fft8xN_real(
	const float t0[restrict static 1],
	const float t4[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f,
	const uint32_t column_count)
{
	float w[8 * column_count];

	fft4xNr(t0, t4, stride_t, row_offset, row_count, w, column_count);

	stuff_real8_needs_to_do_sve(w, column_count, f, stride_f);
}


static inline void sve_ifft8xN_real(
	const float t0[restrict static 1],
	const float t4[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f,
	const uint32_t column_count)
{
	float w[8 * column_count];

	//ifft4xNr(t0, t4, stride_t, row_offset, row_count, w, column_count);

	//stuff_real8_needs_to_do_sve(w, column_count, f, stride_f);
}

static inline void scalar_fft8_real(
	const float t0[restrict static 4],
	const float t4[restrict static 4],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	float w[8];
	sve_fft4_aos(t0, t4, stride_t, row_offset, row_count, w);

	const float half = 0.5f;
	const float g1r = half * (w[2] + w[6]);
	const float g1i = half * (w[3] - w[7]);
	const float two_h1r = w[3] + w[7];
	const float two_h1i = w[6] - w[2];

	const float sqrt2_over_4 = SQRT2_OVER_4;
	const float h1_plus = sqrt2_over_4 * (two_h1i + two_h1r);
	const float h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

	const float f0 = w[0] + w[1];
	const float f4 = w[0] - w[1];
	const float f1r = g1r + h1_plus;
	const float f1i = h1_minus + g1i;
	const float f2r = w[4];
	const float f2i = -w[5];
	const float f3r = g1r - h1_plus;
	const float f3i = h1_minus - g1i;

	/* Store outputs */
	f[0 * stride_f] = f0;
	f[1 * stride_f] = f4;
	f[2 * stride_f] = f1r;
	f[3 * stride_f] = f1i;
	f[4 * stride_f] = f2r;
	f[5 * stride_f] = f2i;
	f[6 * stride_f] = f3r;
	f[7 * stride_f] = f3i;
}

static inline void scalar_fft16_real(
	const float t0[restrict static 8],
	const float t8[restrict static 8],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	float f[restrict static 1],
	size_t stride_f)
{
	float w[16];
	sve_fft8_aos(t0, t8, stride_t, row_offset, row_count, w);

	const float half = 0.5f;
	const float g1r = half * (w[2] + w[14]);
	const float g1i = half * (w[3] - w[15]);
	const float g2r = half * (w[4] + w[12]);
	const float g2i = half * (w[5] - w[13]);
	const float g3r = half * (w[6] + w[10]);
	const float g3i = half * (w[7] - w[11]);

	const float two_h1r = w[3] + w[15];
	const float two_h1i = w[14] - w[2];
	const float two_h2r = w[5] + w[13];
	const float two_h2i = w[12] - w[4];
	const float two_h3r = w[7] + w[11];
	const float two_h3i = w[10] - w[6];

	const float sqrt2_over_4 = SQRT2_OVER_4;
	const float h2_plus = sqrt2_over_4 * (two_h2i + two_h2r);
	const float h2_minus = sqrt2_over_4 * (two_h2i - two_h2r);

	const float half_cos_1pi_over_8 = COS_1PI_OVER_8 * 0.5f;
	const float half_cos_3pi_over_8 = COS_3PI_OVER_8 * 0.5f;

	const float f0 = w[0] + w[1];
	const float f8 = w[0] - w[1];
	const float f1r = g1r + two_h1r * half_cos_1pi_over_8 + two_h1i * half_cos_3pi_over_8;
	const float f1i = g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;
	const float f2r = g2r + h2_plus;
	const float f2i = h2_minus + g2i;
	const float f3r = g3r + two_h3r * half_cos_3pi_over_8 + two_h3i * half_cos_1pi_over_8;
	const float f3i = g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f4r = w[8];
	const float f4i = -w[9];
	const float f5r = g3r - two_h3r * half_cos_3pi_over_8 - two_h3i * half_cos_1pi_over_8;
	const float f5i = -g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f6r = g2r - h2_plus;
	const float f6i = h2_minus - g2i;
	const float f7r = g1r - two_h1r * half_cos_1pi_over_8 - two_h1i * half_cos_3pi_over_8;
	const float f7i = -g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;

	/* Store outputs */
	f[0 * stride_f] = f0;
	f[1 * stride_f] = f8;
	f[2 * stride_f] = f1r;
	f[3 * stride_f] = f1i;
	f[4 * stride_f] = f2r;
	f[5 * stride_f] = f2i;
	f[6 * stride_f] = f3r;
	f[7 * stride_f] = f3i;
	f[8 * stride_f] = f4r;
	f[9 * stride_f] = f4i;
	f[10 * stride_f] = f5r;
	f[11 * stride_f] = f5i;
	f[12 * stride_f] = f6r;
	f[13 * stride_f] = f6i;
	f[14 * stride_f] = f7r;
	f[15 * stride_f] = f7i;
}

static inline void scalar_ifft8_real(
	float f0, float f4, float f1r, float f1i, float f2r, float f2i, float f3r, float f3i,
	float t0[restrict static 4],
	float t4[restrict static 4],
	size_t stride_t)
{
	/* Load inputs and scale */
	const float scale = 0.5f;
	f0 *= scale;
	f4 *= scale;
	f1r *= scale;
	f1i *= scale;
	f3r *= scale;
	f3i *= scale;

	float w[8];

	w[0] = f0 + f4;
	w[1] = f0 - f4;
	w[4] = f2r;
	w[5] = -f2i;

	const float g1r = f1r + f3r;
	const float g1i = f1i - f3i;

	const float h1r = f1r - f3r;
	const float h1i = f1i + f3i;

	const float h1_plus = h1r + h1i;
	const float h1_minus = h1r - h1i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	w[2] = g1r - sqrt2_over2 * h1_plus;
	w[3] = g1i + sqrt2_over2 * h1_minus;
	w[6] = g1r + sqrt2_over2 * h1_plus;
	w[7] = -g1i + sqrt2_over2 * h1_minus;

	sve_ifft4_aos(w, t0, t4, stride_t);
}

static inline void scalar_ifft16_real(
	float f0, float f8, float f1r, float f1i, float f2r, float f2i, float f3r, float f3i,
	float f4r, float f4i, float f5r, float f5i, float f6r, float f6i, float f7r, float f7i,
	float t0[restrict static 8],
	float t8[restrict static 8],
	size_t stride_t)
{
	/* Load inputs and scale */
	const float scale = 0.5f;
	f0 *= scale;
	f8 *= scale;
	f1r *= scale;
	f1i *= scale;
	f2r *= scale;
	f2i *= scale;
	f3r *= scale;
	f3i *= scale;
	f5r *= scale;
	f5i *= scale;
	f6r *= scale;
	f6i *= scale;
	f7r *= scale;
	f7i *= scale;

	float w[16];

	w[0] = f0 + f8;
	w[1] = f0 - f8;
	w[8] = f4r;
	w[9] = -f4i;

	const float g2r = f2r + f6r;
	const float g2i = f2i - f6i;

	const float h2r = f2r - f6r;
	const float h2i = f2i + f6i;

	const float h2_plus = h2r + h2i;
	const float h2_minus = h2r - h2i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	w[4] = g2r - sqrt2_over2 * h2_plus;
	w[5] = g2i + sqrt2_over2 * h2_minus;
	w[12] = g2r + sqrt2_over2 * h2_plus;
	w[13] = -g2i + sqrt2_over2 * h2_minus;

	const float g1r = f1r + f7r;
	const float g1i = f1i - f7i;
	const float g3r = f3r + f5r;
	const float g3i = f3i - f5i;

	const float h1r = f1r - f7r;
	const float h1i = f1i + f7i;
	const float h3r = f3r - f5r;
	const float h3i = f3i + f5i;

	const float cos_1pi_over_8 = COS_1PI_OVER_8;
	const float cos_3pi_over_8 = COS_3PI_OVER_8;
	w[2] = g1r - h1i * cos_1pi_over_8 - h1r * cos_3pi_over_8;
	w[3] = g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;
	w[14] = g1r + h1i * cos_1pi_over_8 + h1r * cos_3pi_over_8;
	w[15] = -g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;

	w[6] = g3r - h3i * cos_3pi_over_8 - h3r * cos_1pi_over_8;
	w[7] = g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;
	w[10] = g3r + h3i * cos_3pi_over_8 + h3r * cos_1pi_over_8;
	w[11] = -g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;

	sve_ifft8_aos(w, t0, t8, stride_t);
}