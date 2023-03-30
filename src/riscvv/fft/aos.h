#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>


static inline void riscvv_fft4_aos(
	const float t_lo[restrict static 4],
	const float t_hi[restrict static 4],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f[restrict static 8])
{
	/* Load inputs and FFT4: butterfly */
	/* Load inputs and FFT4: butterfly */
	float w[8] = {0.0f};

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w[0] = w[4] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 4 && row_end > 4) {
		w[4] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[0], &w[4]);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 1) {
		w[1] = w[5] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 5 && row_end > 5) {
		w[5] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[1], &w[5]);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 2) {
		w[2] = w[6] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 6 && row_end > 6) {
		w[6] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[2], &w[6]);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 3) {
		w[3] = w[7] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 7 && row_end > 7) {
		w[7] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[3], &w[7]);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

fft4_twiddle:
	;
	/*
	 * FFT4: multiplication by twiddle factors:
	 *   w3r, w3i = w3i, -w3r
	 * (negation of w3i is merged into the next butterfly)
	 */
	int indices_a[8] = {0 * sizeof(float), 1 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float)};
	int indices_b[8] = {2 * sizeof(float), 3 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float)};
	float mul_b[8] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

	// Granted vector length
	long gvl = __builtin_epi_vsetvl(7, __epi_e32, __epi_m1);
	for(uint32_t i = 0; i < 8; i += gvl) {
		__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
		__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(w, v_indices_a, gvl);

		__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
		__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(w, v_indices_b, gvl);

		__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

		// b * mul_b + a
		__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
		
		__builtin_epi_vstore_2xf32(f + i, v_final, gvl);
	}
}

static inline void riscvv_ifft4_aos(
	float w[restrict static 8],
	float t0[restrict static 4],
	float t2[restrict static 4],
	size_t stride_t)
{
	// * 2x IFFT2: butterfly
	// * IFFT4: multiplication by twiddle factors
	// * IFFT4: scaling by 1/4 
	long gvl = __builtin_epi_vsetvl(7, __epi_e32, __epi_m1);
	__epi_2xf32 v_quarter = __builtin_epi_vfmv_v_f_2xf32(0.25f, gvl);
	float part1_w[8] = {0.0};
	{
		uint32_t indices_a[8] = {0 * sizeof(float), 1 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 3 * sizeof(float), 2 * sizeof(float)};
		uint32_t indices_b[8] = {4 * sizeof(float), 5 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float)};
		float mul_b[8] = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};

		for(uint32_t  i=0; i < 8; i+=gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			v_final = __builtin_epi_vfmul_2xf32(v_final, v_quarter, gvl);
			
			__builtin_epi_vstore_2xf32(part1_w + i, v_final, gvl);
		}
	}

	/* IFFT4: butterfly and store outputs */

	float part2_w[8] = {0.0};
	{
		uint32_t indices_a[8] = {0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float)};
		uint32_t indices_b[8] = {4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float)};
		float mul_b[8] = {1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

		for(uint32_t  i=0; i < 8; i+=gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			
			__builtin_epi_vstore_2xf32(part2_w + i, v_final, gvl);
		}
	}
	
	// Store output to t0 and t2
	gvl = __builtin_epi_vsetvl(3, __epi_e32, __epi_m1);
	{
		for(uint32_t  i=0; i < 4; i+=gvl){
			__epi_2xf32 w_low = __builtin_epi_vload_2xf32(part2_w + i, gvl);
			__epi_2xf32 w_high = __builtin_epi_vload_2xf32(part2_w + 4 + i, gvl);
			__builtin_epi_vstore_strided_2xf32(t0, w_low, stride_t * sizeof(float), gvl);
			__builtin_epi_vstore_strided_2xf32(t2, w_high, stride_t * sizeof(float), gvl);
			t0 += stride_t * gvl;
			t2 += stride_t * gvl;
		}
	}
}

static inline void riscvv_fft8_aos(
	const float t_lo[restrict static 8],
	const float t_hi[restrict static 8],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f[restrict static 16]
)
{
	/* Load inputs and FFT8: butterfly */
	float w[16] = {0.0f};

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w[0] = w[8] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 8 && row_end > 8) {
		w[8] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[0], &w[8]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 1) {
		w[1] = w[9] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 9 && row_end > 9) {
		w[9] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[1], &w[9]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 2) {
		w[2] = w[10] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 10 && row_end > 10) {
		w[10] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[2], &w[10]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 3) {
		w[3] = w[11] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 11 && row_end > 11) {
		w[11] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[3], &w[11]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 4) {
		w[4] = w[12] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 12 && row_end > 12) {
		w[12] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[4], &w[12]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 5) {
		w[5] = w[13] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 13 && row_end > 13) {
		w[13] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[5], &w[13]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 6) {
		w[6] = w[14] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 14 && row_end > 14) {
		w[14] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[6], &w[14]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 7) {
		w[7] = w[15] = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 15 && row_end > 15) {
		w[15] = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w[7], &w[15]);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

fft8_twiddle:;
	/*
	 * FFT8: multiplication by twiddle factors:
	 *   
	 *   w5r, w5i = sqrt(2)/2 (w5i + w5r),  sqrt(2)/2 (w5i - w5r)
	 *   w6r, w6i = w6i, -w6r
	 *   w7r, w7i = sqrt(2)/2 (w7i - w7r), -sqrt(2)/2 (w7i + w7r)
	 *
	 * (negation of w6i and w7i is merged into the next butterfly)
	 */
	const float sqrt2_over_2 = SQRT2_OVER_2;
	const float new_w5r = sqrt2_over_2 * (w[11] + w[10]);
	const float new_w5i = sqrt2_over_2 * (w[11] - w[10]);
	const float new_w7r = sqrt2_over_2 * (w[15] - w[14]);
	const float minus_new_w7i = sqrt2_over_2 * (w[15] + w[14]);
	w[10] = new_w5r;
	w[11] = new_w5i;
	scalar_swap(&w[12], &w[13]);
	w[14] = new_w7r;
	w[15] = minus_new_w7i;

	long gvl = __builtin_epi_vsetvl(15, __epi_e32, __epi_m1);
	float part1_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float)};
		uint32_t indices_b[16] = {4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float)};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

		for(uint32_t  i=0; i < 16; i+=gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			
			__builtin_epi_vstore_2xf32(part1_w + i, v_final, gvl);
		}
	}

	/*
	 * 2x FFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = w3i, -w3r
	 *   w7r, w7i = w7i, -w7r
	 *
	 * (negation of w3i and w7i is merged into the next butterfly)
	 */
	/* Bit reversal */
	/* 4x FFT2: butterfly*/

	{
		uint32_t indices_a[16] = {0 * sizeof(float), 1 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float)};
		uint32_t indices_b[16] = {2 * sizeof(float), 3 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 15 * sizeof(float), 14 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 15 * sizeof(float), 14 * sizeof(float)};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

		for(uint32_t  i=0; i < 16; i += gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			
			__builtin_epi_vstore_2xf32(f + i, v_final, gvl);
		}
	}
}

static inline void riscvv_ifft8_aos(
	float w[restrict static 16],
	float t_lo[restrict static 8],
	float t_hi[restrict static 8],
	size_t stride_t)
{
	/* Bit reversal */
	/* 4x IFFT2: butterfly */
	long gvl = __builtin_epi_vsetvl(15, __epi_e32, __epi_m1);
	float part1_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0 * sizeof(float), 1 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float)};
		uint32_t indices_b[16] = {8 * sizeof(float), 9 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float)};
		float mul_b[16] = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};

		for(uint32_t  i=0; i < 16; i += gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			
			__builtin_epi_vstore_2xf32(part1_w + i, v_final, gvl);
		}
	}

	/*
	 * 2x IFFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = -w3i, w3r
	 *   w7r, w7i = -w7i, w7r
	 *
	 * (negation of w3r and w7r is merged into the next butterfly)
	 */
	/*
	 * 2x IFFT4: butterfly
	 */
	float part2_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float)};
		uint32_t indices_b[16] = {4 * sizeof(float), 5 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 7 * sizeof(float), 6 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 15 * sizeof(float), 14 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 15 * sizeof(float), 14 * sizeof(float)};
		float mul_b[16] = {1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

		for(uint32_t  i=0; i < 16; i+= gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(part1_w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			
			__builtin_epi_vstore_2xf32(part2_w + i, v_final, gvl);
		}
	}

	/*
	 * IFFT8: multiplication by twiddle factors and scaling by 1/8: (scaling moved to part 3)
	 *
	 *   w5r, w5i =  sqrt(2)/2 (w5r - w5i), sqrt(2)/2 (w5r + w5i)
	 *   w6r, w6i = -w6i, w6r
	 *   w7r, w7i = -sqrt(2)/2 (w7r + w7i), sqrt(2)/2 (w7r - w7i)
	 *
	 * (negation of w6r and w7r is merged into the next butterfly)
	 */
	const float sqrt2_over_2 = SQRT2_OVER_2;
	const float new_w5r = sqrt2_over_2 * (part2_w[10] - part2_w[11]);
	const float new_w5i = sqrt2_over_2 * (part2_w[10] + part2_w[11]);
	const float minus_new_w7r = sqrt2_over_2 * (part2_w[14] + part2_w[15]);
	const float new_w7i = sqrt2_over_2 * (part2_w[14] + part2_w[15]);
	part2_w[10] = new_w5r;
	part2_w[11] = new_w5i;
	scalar_swap(&part2_w[12], &part2_w[13]);
	part2_w[14] = minus_new_w7r;
	part2_w[15] = new_w7i;

	/* IFFT8: butterfly */
	/*
	scalar_butterfly(&w0r, &w4r);
	scalar_butterfly(&w0i, &w4i);
	scalar_butterfly(&w1r, &w5r);
	scalar_butterfly(&w1i, &w5i);
	scalar_butterfly_with_negated_b(&w2r, &w6r);
	scalar_butterfly(&w2i, &w6i);
	scalar_butterfly_with_negated_b(&w3r, &w7r);
	scalar_butterfly(&w3i, &w7i);
	*/
	float part3_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float), 0 * sizeof(float), 1 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 4 * sizeof(float), 5 * sizeof(float), 6 * sizeof(float), 7 * sizeof(float)};
		uint32_t indices_b[16] = {8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float), 8 * sizeof(float), 9 * sizeof(float), 10 * sizeof(float), 11 * sizeof(float), 12 * sizeof(float), 13 * sizeof(float), 14 * sizeof(float), 15 * sizeof(float)};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
		__epi_2xf32 v_eighth = __builtin_epi_vfmv_v_f_2xf32(0.125f, gvl);

		for(uint32_t  i=0; i < 16; i+= gvl){
			__epi_2xi32 v_indices_a = __builtin_epi_vload_2xi32(indices_a + i, gvl);
			__epi_2xf32 v_a = __builtin_epi_vload_indexed_2xf32(part2_w, v_indices_a, gvl);

			__epi_2xi32 v_indices_b = __builtin_epi_vload_2xi32(indices_b + i, gvl);
			__epi_2xf32 v_b = __builtin_epi_vload_indexed_2xf32(part2_w, v_indices_b, gvl);

			__epi_2xf32 v_mul_b = __builtin_epi_vload_2xf32(mul_b + i, gvl);

			// b * mul_b + a
			__epi_2xf32 v_final = __builtin_epi_vfmacc_2xf32(v_a, v_b, v_mul_b, gvl);
			v_final = __builtin_epi_vfmul_2xf32(v_final, v_eighth, gvl);
			
			__builtin_epi_vstore_2xf32(part3_w + i, v_final, gvl);
		}
	}

	// Store outputs
	gvl = __builtin_epi_vsetvl(7, __epi_e32, __epi_m1);
	{
		for(uint32_t  i = 0; i < 8; i += gvl){
			__epi_2xf32 w_low = __builtin_epi_vload_2xf32(part3_w + i, gvl);
			__epi_2xf32 w_high = __builtin_epi_vload_2xf32(part3_w + 8 + i, gvl);
			__builtin_epi_vstore_strided_2xf32(t_lo, w_low, stride_t * sizeof(float), gvl);
			__builtin_epi_vstore_strided_2xf32(t_hi, w_high, stride_t * sizeof(float), gvl);
			t_lo += stride_t * gvl;
			t_hi += stride_t * gvl;
		}
	}
}
