#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>
#include <arm_sve.h>

static inline void sve_fft4_aos(
	const float t_lo[restrict static 4],
	const float t_hi[restrict static 4],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f0r[restrict static 1],
	float f0i[restrict static 1],
	float f1r[restrict static 1],
	float f1i[restrict static 1],
	float f2r[restrict static 1],
	float f2i[restrict static 1],
	float f3r[restrict static 1],
	float f3i[restrict static 1])
{
	/* Load inputs and FFT4: butterfly */
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i;
	w0r = w0i = w1r = w1i = w2r = w2i = w3r = w3i = 0.0f;

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w0r = w2r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 4 && row_end > 4) {
		w2r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w0r, &w2r);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 1) {
		w0i = w2i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 5 && row_end > 5) {
		w2i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w0i, &w2i);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 2) {
		w1r = w3r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 6 && row_end > 6) {
		w3r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w1r, &w3r);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

	if (row_start <= 3) {
		w1i = w3i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}
	if (row_start <= 7 && row_end > 7) {
		w3i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w1i, &w3i);
		if (--row_count == 0) {
			goto fft4_twiddle;
		}
	}

fft4_twiddle:
	uint32_t indices_a[8] = {0, 1, 4, 5, 0, 1, 4, 5};
	uint32_t indices_b[8] = {2, 3, 7, 6, 2, 3, 7, 6};
	float mul_b[8] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
	float new_w[8] = {0.0};
	float old_w[8] = {w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i};

	svbool_t pq;
	svfloat32_t sv_a, sv_b;

	uint64_t vector_len = svlen_f32(sv_a);
	for(uint32_t  i=0; i < 8; i+=vector_len){
		pq = svwhilelt_b32_s32(i, 8); 

		svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
		sv_a = svld1_gather_index(pq, old_w, sv_indexes_a);

		svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
		sv_b = svld1_gather_index(pq, old_w, sv_indexes_b);

		svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
		sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);

		svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
		svst1(pq, new_w, sv_added);
	}

	*f0r = new_w[0];
	*f0i = new_w[1];
	*f1r = new_w[2];
	*f1i = new_w[3];
	*f2r = new_w[4];
	*f2i = new_w[5];
	*f3r = new_w[6];
	*f3i = new_w[7];
}

static inline void sve_ifft4_aos(
	float w0r, float w0i, float w1r, float w1i, float w2r, float w2i, float w3r, float w3i,
	float t0[restrict static 4],
	float t2[restrict static 4],
	size_t stride_t)
{

	// * 2x IFFT2: butterfly
	// * IFFT4: multiplication by twiddle factors
	// * IFFT4: scaling by 1/4 

	svfloat32_t sv_quarter = svdup_f32(0.25f);
	uint64_t vector_len = svlen_f32(sv_quarter);
	float part1_w[8] = {0.0};

	{
		uint32_t indices_a[8] = {0, 1, 0, 1, 2, 3, 3, 2};
		uint32_t indices_b[8] = {4, 5, 4, 5, 6, 7, 7, 6};
		float mul_b[8] = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
		float old_w[8] = {w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i};

		for(uint32_t  i=0; i < 8; i+=vector_len){
			svbool_t pq = svwhilelt_b32_s32(i, 8); 

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			svfloat32_t sv_a = svld1_gather_index(pq, old_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			svfloat32_t sv_b = svld1_gather_index(pq, old_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);

			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svfloat32_t sv_final = svmul_f32_m(pq, sv_added, sv_quarter);
			
			svst1(pq, part1_w, sv_final);
		}
	}

	/* IFFT4: butterfly and store outputs */

	float part2_w[8] = {0.0};
	
	{
		uint32_t indices_a[8] = {0, 1, 2, 3, 0, 1, 2, 3};
		uint32_t indices_b[8] = {4, 5, 6, 7, 4, 5, 6, 7};
		float mul_b[8] = {1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

		for(uint32_t  i=0; i < 8; i+=vector_len){
			svbool_t pq = svwhilelt_b32_s32(i, 8); 

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			svfloat32_t sv_a = svld1_gather_index(pq, part1_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			svfloat32_t sv_b = svld1_gather_index(pq, part1_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);

			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			
			svst1(pq, part2_w, sv_added);
		}
	}
	
	// Store output to t0 and t2

	{
		svint32_t offsets = svindex_s32(0, stride_t * sizeof(float));
		for(uint32_t  i=0; i < 4; i+=vector_len){
			svbool_t pq = svwhilelt_b32_s32(i, 4); 
			svfloat32_t w_low = svld1(pq, part2_w + i * vector_len);
			svfloat32_t w_high = svld1(pq, part2_w + 4 + i * vector_len);
			svst1_scatter_offset(pq, t0, offsets, w_low);
			svst1_scatter_offset(pq, t2, offsets, w_high);
			t0 += stride_t * vector_len;
			t2 += stride_t * vector_len;
		}
	}
}

static inline void scalar_fft8_aos(
	const float t_lo[restrict static 8],
	const float t_hi[restrict static 8],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f0r[restrict static 1],
	float f0i[restrict static 1],
	float f1r[restrict static 1],
	float f1i[restrict static 1],
	float f2r[restrict static 1],
	float f2i[restrict static 1],
	float f3r[restrict static 1],
	float f3i[restrict static 1],
	float f4r[restrict static 1],
	float f4i[restrict static 1],
	float f5r[restrict static 1],
	float f5i[restrict static 1],
	float f6r[restrict static 1],
	float f6i[restrict static 1],
	float f7r[restrict static 1],
	float f7i[restrict static 1])
{
	/* Load inputs and FFT8: butterfly */
	float w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i;
	w0r = w0i = w1r = w1i = w2r = w2i = w3r = w3i = w4r = w4i = w5r = w5i = w6r = w6i = w7r = w7i = 0.0f;

	const uint32_t row_end = row_start + row_count;
	if (row_start <= 0) {
		w0r = w4r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 8 && row_end > 8) {
		w4r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w0r, &w4r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 1) {
		w0i = w4i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 9 && row_end > 9) {
		w4i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w0i, &w4i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 2) {
		w1r = w5r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 10 && row_end > 10) {
		w5r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w1r, &w5r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 3) {
		w1i = w5i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 11 && row_end > 11) {
		w5i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w1i, &w5i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 4) {
		w2r = w6r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 12 && row_end > 12) {
		w6r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w2r, &w6r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 5) {
		w2i = w6i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 13 && row_end > 13) {
		w6i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w2i, &w6i);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 6) {
		w3r = w7r = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 14 && row_end > 14) {
		w7r = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w3r, &w7r);
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}

	if (row_start <= 7) {
		w3i = w7i = *t_lo;
		t_lo += stride_t;
		if (--row_count == 0) {
			goto fft8_twiddle;
		}
	}
	if (row_start <= 15 && row_end > 15) {
		w7i = *t_hi;
		t_hi += stride_t;
		scalar_butterfly(&w3i, &w7i);
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
	const float new_w5r = sqrt2_over_2 * (w5i + w5r);
	const float new_w5i = sqrt2_over_2 * (w5i - w5r);
	const float new_w7r = sqrt2_over_2 * (w7i - w7r);
	const float minus_new_w7i = sqrt2_over_2 * (w7i + w7r);
	w5r = new_w5r;
	w5i = new_w5i;
	scalar_swap(&w6r, &w6i);
	w7r = new_w7r;
	w7i = minus_new_w7i;

	/*
	 * 2x FFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);
	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly_with_negated_b(&w4i, &w6i);
	scalar_butterfly(&w5r, &w7r);
	scalar_butterfly_with_negated_b(&w5i, &w7i);

	/*
	 * 2x FFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = w3i, -w3r
	 *   w7r, w7i = w7i, -w7r
	 *
	 * (negation of w3i and w7i is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);

	/*
	 * 4x FFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly_with_negated_b(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly_with_negated_b(&w6i, &w7i);

	/* Bit reversal */
	scalar_swap(&w1r, &w4r);
	scalar_swap(&w1i, &w4i);
	scalar_swap(&w3r, &w6r);
	scalar_swap(&w3i, &w6i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
	*f4r = w4r;
	*f4i = w4i;
	*f5r = w5r;
	*f5i = w5i;
	*f6r = w6r;
	*f6i = w6i;
	*f7r = w7r;
	*f7i = w7i;
}

static inline void scalar_ifft8_aos(
	float w0r, float w0i, float w1r, float w1i, float w2r, float w2i, float w3r, float w3i,
	float w4r, float w4i, float w5r, float w5i, float w6r, float w6i, float w7r, float w7i,
	float t_lo[restrict static 8],
	float t_hi[restrict static 8],
	size_t stride_t)
{
	/* Bit reversal */
	scalar_swap(&w1r, &w4r);
	scalar_swap(&w1i, &w4i);
	scalar_swap(&w3r, &w6r);
	scalar_swap(&w3i, &w6i);

	/*
	 * 4x IFFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly(&w6i, &w7i);

	/*
	 * 2x IFFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = -w3i, w3r
	 *   w7r, w7i = -w7i, w7r
	 *
	 * (negation of w3r and w7r is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);

	/*
	 * 2x IFFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly_with_negated_b(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);
	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly(&w4i, &w6i);
	scalar_butterfly_with_negated_b(&w5r, &w7r);
	scalar_butterfly(&w5i, &w7i);

	/*
	 * IFFT8: multiplication by twiddle factors and scaling by 1/8:
	 *
	 *   w5r, w5i =  sqrt(2)/2 (w5r - w5i), sqrt(2)/2 (w5r + w5i)
	 *   w6r, w6i = -w6i, w6r
	 *   w7r, w7i = -sqrt(2)/2 (w7r + w7i), sqrt(2)/2 (w7r - w7i)
	 *
	 * (negation of w6r and w7r is merged into the next butterfly)
	 */
	const float sqrt2_over_2 = SQRT2_OVER_2 * 0.125f;
	const float new_w5r = sqrt2_over_2 * (w5r - w5i);
	const float new_w5i = sqrt2_over_2 * (w5r + w5i);
	const float minus_new_w7r = sqrt2_over_2 * (w7r + w7i);
	const float new_w7i = sqrt2_over_2 * (w7r - w7i);
	w5r = new_w5r;
	w5i = new_w5i;
	scalar_swap(&w6r, &w6i);
	w7r = minus_new_w7r;
	w7i = new_w7i;

	/* IFFT8: scaling of remaining coefficients by 1/8 */
	const float scale = 0.125f;
	w0r *= scale;
	w0i *= scale;
	w1r *= scale;
	w1i *= scale;
	w2r *= scale;
	w2i *= scale;
	w3r *= scale;
	w3i *= scale;
	w4r *= scale;
	w4i *= scale;
	w6r *= scale;
	w6i *= scale;

	/* IFFT8: butterfly and store outputs */
	scalar_butterfly(&w0r, &w4r);
	*t_lo = w0r;
	t_lo += stride_t;
	*t_hi = w4r;
	t_hi += stride_t;

	scalar_butterfly(&w0i, &w4i);
	*t_lo = w0i;
	t_lo += stride_t;
	*t_hi = w4i;
	t_hi += stride_t;

	scalar_butterfly(&w1r, &w5r);
	*t_lo = w1r;
	t_lo += stride_t;
	*t_hi = w5r;
	t_hi += stride_t;

	scalar_butterfly(&w1i, &w5i);
	*t_lo = w1i;
	t_lo += stride_t;
	*t_hi = w5i;
	t_hi += stride_t;

	scalar_butterfly_with_negated_b(&w2r, &w6r);
	*t_lo = w2r;
	t_lo += stride_t;
	*t_hi = w6r;
	t_hi += stride_t;

	scalar_butterfly(&w2i, &w6i);
	*t_lo = w2i;
	t_lo += stride_t;
	*t_hi = w6i;
	t_hi += stride_t;

	scalar_butterfly_with_negated_b(&w3r, &w7r);
	*t_lo = w3r;
	t_lo += stride_t;
	*t_hi = w7r;
	t_hi += stride_t;

	scalar_butterfly(&w3i, &w7i);
	*t_lo = w3i;
	*t_hi = w7i;
}
