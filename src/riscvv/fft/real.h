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
	const __epi_2xi32 ind_store = indexA((uint32_t []){0, 4, 8, 12}, 4, 8 * 4, gvl);

	// Offsets and masks

    const uint32_t row_end = row_start + row_count;
    const bool a_pred[4] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3};
	__epi_2xi32 t_lo_offset_r = aos4_offset_r(&a_pred, stride_t, gvl);
	__epi_2xi32 t_lo_offset_i = aos4_offset_i(&a_pred, stride_t, gvl);

    const bool b_pred[4] = {row_start <= 4 && row_end > 4, row_start <= 5 && row_end > 5, row_start <= 6 && row_end > 6, row_start <= 7 && row_end > 7};
	__epi_2xi32 t_hi_offset_r = aos4_offset_r(&b_pred, stride_t, gvl);
	__epi_2xi32 t_hi_offset_i = aos4_offset_i(&b_pred, stride_t, gvl);

    bool no_jump[8];
	jump_arr(&no_jump, &a_pred, &b_pred);

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
		__epi_2xi32 new_a_r = butterfly_add(a, b, gvl);
		__epi_2xi32 new_a_i = butterfly_add(a, b, gvl);
		__epi_2xi32 new_b_r = butterfly_sub(a, b, gvl);
		__epi_2xi32 new_b_i = butterfly_sub(a, b, gvl);

		//__epi_2xf32 new_bt = cmulc_twiddle(new_b, twiddle, gvl);
		__epi_2xf32 new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);
		__epi_2xf32 new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);

		shuffle(&new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b, gvl);

		// stage2
		new_a = butterfly_add(a, b, gvl);
		new_b = butterfly_sub(a, b, gvl);

		// store
		svst1_scatter_offset(pg, f + i * 2 + 0, ind_store, new_a);
		svst1_scatter_offset(pg, f + i * 2 + 4, ind_store, new_b);

		i += gvl;
	}
}

static inline void riscvv_fft8xN_real(
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

	stuff_for_fft8x8_sve(w, column_count, f, stride_f);
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
	riscvv_fft4_aos(t0, t4, stride_t, row_offset, row_count, w);

	const float half = 0.5f;
	const float g1r = half * (w[2] + w[6]);
	const float g1i = half * (w[3] - w[7]);
	const float two_h1r = w[3] + w[7];
	const float two_h1i = w[6] - w[2];

	const float sqrt2_over_4 = SQRT2_OVER_4;
	const float h1_plus  = sqrt2_over_4 * (two_h1i + two_h1r);
	const float h1_minus = sqrt2_over_4 * (two_h1i - two_h1r);

	const float f0 = w[0] + w[1];
	const float f4 = w[0] - w[1];
	const float f1r = g1r + h1_plus;
	const float f1i = h1_minus + g1i;
	const float f2r =  w[4];
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
	riscvv_fft8_aos(t0, t8, stride_t, row_offset, row_count, w);

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
	const float h2_plus  = sqrt2_over_4 * (two_h2i + two_h2r);
	const float h2_minus = sqrt2_over_4 * (two_h2i - two_h2r);

	const float half_cos_1pi_over_8 = COS_1PI_OVER_8 * 0.5f;
	const float half_cos_3pi_over_8 = COS_3PI_OVER_8 * 0.5f;

	const float f0  =  w[0] + w[1];
	const float f8  =  w[0] - w[1];
	const float f1r =  g1r + two_h1r * half_cos_1pi_over_8 + two_h1i * half_cos_3pi_over_8;
	const float f1i =  g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;
	const float f2r =  g2r + h2_plus;
	const float f2i =  h2_minus + g2i;
	const float f3r =  g3r + two_h3r * half_cos_3pi_over_8 + two_h3i * half_cos_1pi_over_8;
	const float f3i =  g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f4r =  w[8];
	const float f4i = -w[9];
	const float f5r =  g3r - two_h3r * half_cos_3pi_over_8 - two_h3i * half_cos_1pi_over_8;
	const float f5i = -g3i + two_h3i * half_cos_3pi_over_8 - two_h3r * half_cos_1pi_over_8;
	const float f6r =  g2r - h2_plus;
	const float f6i =  h2_minus - g2i;
	const float f7r =  g1r - two_h1r * half_cos_1pi_over_8 - two_h1i * half_cos_3pi_over_8;
	const float f7i = -g1i + two_h1i * half_cos_1pi_over_8 - two_h1r * half_cos_3pi_over_8;

	/* Store outputs */
	f[ 0 * stride_f] = f0;
	f[ 1 * stride_f] = f8;
	f[ 2 * stride_f] = f1r;
	f[ 3 * stride_f] = f1i;
	f[ 4 * stride_f] = f2r;
	f[ 5 * stride_f] = f2i;
	f[ 6 * stride_f] = f3r;
	f[ 7 * stride_f] = f3i;
	f[ 8 * stride_f] = f4r;
	f[ 9 * stride_f] = f4i;
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
	f0  *= scale;
	f4  *= scale;
	f1r *= scale;
	f1i *= scale;
	f3r *= scale;
	f3i *= scale;

	float w[8];

	w[0] =  f0 + f4;
	w[1] =  f0 - f4;
	w[4] =  f2r;
	w[5] = -f2i;

	const float g1r = f1r + f3r;
	const float g1i = f1i - f3i;

	const float h1r = f1r - f3r;
	const float h1i = f1i + f3i;

	const float h1_plus  = h1r + h1i;
	const float h1_minus = h1r - h1i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	w[2] =  g1r - sqrt2_over2 * h1_plus;
	w[3] =  g1i + sqrt2_over2 * h1_minus;
	w[6] =  g1r + sqrt2_over2 * h1_plus;
	w[7] = -g1i + sqrt2_over2 * h1_minus;

	riscvv_ifft4_aos(w, t0, t4, stride_t);
}

static inline void scalar_ifft16_real(
	float f0,  float f8,  float f1r, float f1i, float f2r, float f2i, float f3r, float f3i,
	float f4r, float f4i, float f5r, float f5i, float f6r, float f6i, float f7r, float f7i,
	float t0[restrict static 8],
	float t8[restrict static 8],
	size_t stride_t)
{
	/* Load inputs and scale */
	const float scale = 0.5f;
	f0  *= scale;
	f8  *= scale;
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

	w[0] =  f0 + f8;
	w[1] =  f0 - f8;
	w[8] =  f4r;
	w[9] = -f4i;

	const float g2r = f2r + f6r;
	const float g2i = f2i - f6i;

	const float h2r = f2r - f6r;
	const float h2i = f2i + f6i;

	const float h2_plus  = h2r + h2i;
	const float h2_minus = h2r - h2i;

	const float sqrt2_over2 = SQRT2_OVER_2;
	w[4] =  g2r - sqrt2_over2 * h2_plus;
	w[5] =  g2i + sqrt2_over2 * h2_minus;
	w[12] =  g2r + sqrt2_over2 * h2_plus;
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
	w[2] =  g1r - h1i * cos_1pi_over_8 - h1r * cos_3pi_over_8;
	w[3] =  g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;
	w[14] =  g1r + h1i * cos_1pi_over_8 + h1r * cos_3pi_over_8;
	w[15] = -g1i + h1r * cos_1pi_over_8 - h1i * cos_3pi_over_8;

	w[6] =  g3r - h3i * cos_3pi_over_8 - h3r * cos_1pi_over_8;
	w[7] =  g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;
	w[10] =  g3r + h3i * cos_3pi_over_8 + h3r * cos_1pi_over_8;
	w[11] = -g3i + h3r * cos_3pi_over_8 - h3i * cos_1pi_over_8;

	riscvv_ifft8_aos(w, t0, t8, stride_t);
}