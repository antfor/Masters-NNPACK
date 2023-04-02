#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>
#include <arm_sve.h>
#include <stdbool.h>
#include <sve/fft/fft-util.h>


inline static void aos4_offset(const bool *n, size_t stride, svuint32_t *t_n_offset)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

    *t_n_offset = index4(0, cSum_n[0] * stride * 4, cSum_n[1] * stride * 4, cSum_n[2] * stride * 4, 1 * 4);
}

inline static void aos4_pred_and_offset(uint32_t row_start, uint32_t row_count, svbool_t *pg_a, svbool_t *pg_b, size_t stride, svuint32_t *t_lo_offset, svuint32_t *t_hi_offset)
{
    const uint32_t row_end = row_start + row_count;

    const bool a[4] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3};
    const bool b[4] = {row_start <= 4 && row_end > 4, row_start <= 5 && row_end > 5, row_start <= 6 && row_end > 6, row_start <= 7 && row_end > 7};

    bool no_jump[8];
    no_jump[0] = 1;
    no_jump[1] = no_jump[0] && !(a[0] && --row_count == 0);
    no_jump[2] = no_jump[1] && !(b[0] && --row_count == 0);
    no_jump[3] = no_jump[2] && !(a[1] && --row_count == 0);
    no_jump[4] = no_jump[3] && !(b[1] && --row_count == 0);
    no_jump[5] = no_jump[4] && !(a[2] && --row_count == 0);
    no_jump[6] = no_jump[5] && !(b[2] && --row_count == 0);
    no_jump[7] = no_jump[6] && !(a[3] && --row_count == 0);

    *pg_a = svdupq_b32(a[0] && no_jump[0], a[1] && no_jump[2], a[2] && no_jump[4], a[3] && no_jump[6]);
    *pg_b = svdupq_b32(b[0] && no_jump[1], b[1] && no_jump[3], b[2] && no_jump[5], b[3] && no_jump[7]);

    aos4_offset(a, stride, t_lo_offset);
    aos4_offset(b, stride, t_hi_offset);
}

// todo remove scalar

static inline void sve_fft8_aos(
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

	/*
	 * 2x FFT4: butterfly
	 */

	/*
		w = [
			0 : 0 + 4,
			1 : 1 + 5,
			2 : 2 + 6,
			3 : 3 + 7,
			4 : 0 - 4,
			5 : 1 - 5,
			6 : 2 - 6,
			7 : 3 - 7,
			8 : 8 + 12,
			9 : 9 - 13, 
			10: 10 + 14,
			11: 11 - 15,
			12: 8 - 12,
			13: 9 + 13,
			14: 10 - 14,
			15: 11 + 15,
		]	
	*/

	float part1_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11};
		uint32_t indices_b[16] = {4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indices_a = svld1(pq, indices_a + i);
			sv_a = svld1_gather_index(pq, w, sv_indices_a);

			svuint32_t sv_indices_b = svld1(pq, indices_b + i);
			sv_b = svld1_gather_index(pq, w, sv_indices_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i);
			
			svfloat32_t sv_final = svmad_f32_m(pq, sv_b, sv_mul_b, sv_a);

			svst1(pq, part1_w, sv_final);
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

	/*
		w = [
			0  w0r = 0 + 2
			1  w0i = 1 + 3
			2  w1r = 8 + 10
			3  w1i = 9 + 11
			4  w2r = 4 + 7
			5  w2i = 5 - 6
			6  w3r = 12 + 15
			7  w3i = 13 - 14
			8  w4r = 0 - 2
			9  w4i = 1 - 3
			10 w5r = 8 - 10
			11 w5i = 9 - 11
			12 w6r = 4 - 7
			13 w6i = 5 + 6
			14 w7r = 12 - 15
			15 w7i = 13 + 14
		]	
	*/
	float part2_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0, 1, 8, 9, 4, 5, 12, 13, 0, 1, 8, 9, 4, 5, 12, 13};
		uint32_t indices_b[16] = {2, 3, 10, 11, 7, 6, 15, 14, 2, 3, 10, 11, 7, 6, 15, 14};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indices_a = svld1(pq, indices_a + i);
			sv_a = svld1_gather_index(pq, part1_w, sv_indices_a);

			svuint32_t sv_indices_b = svld1(pq, indices_b + i);
			sv_b = svld1_gather_index(pq, part1_w, sv_indices_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i);
			
			svfloat32_t sv_final = svmad_f32_m(pq, sv_b, sv_mul_b, sv_a);

			svst1(pq, f, sv_final);
		}
	}
}

static inline void sve_ifft8_aos(
	float w[restrict static 16],
	float t_lo[restrict static 8],
	float t_hi[restrict static 8],
	size_t stride_t)
{
	/* Bit reversal */
	/* 4x IFFT2: butterfly */
	/*
		w = [
			0  w0r = 0 + 8
			1  w0i = 1 + 9
			2  w1r = 0 - 8
			3  w1i = 1 - 9
			4  w2r = 4 + 12
			5  w2i = 5 + 13
			6  w3r = 4 - 12
			7  w3i = 5 - 13
			8  w4r = 2 + 10
			9  w4i = 3 + 11
			10 w5r = 2 - 10
			11 w5i = 3 - 11
			12 w6r = 6 + 14
			13 w6i = 7 + 15
			14 w7r = 6 - 14
			15 w7i = 7 - 15
		]	
	*/

	float part1_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0, 1, 0, 1, 4, 5, 4, 5, 2, 3, 2, 3, 6, 7, 6, 7};
		uint32_t indices_b[16] = {8, 9, 8, 9, 12, 13, 12, 13, 10, 11, 10, 11, 14, 15, 14, 15};
		float mul_b[16] = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indices_a = svld1(pq, indices_a + i);
			sv_a = svld1_gather_index(pq, w, sv_indices_a);

			svuint32_t sv_indices_b = svld1(pq, indices_b + i);
			sv_b = svld1_gather_index(pq, w, sv_indices_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i);
			
			svfloat32_t sv_final = svmad_f32_m(pq, sv_b, sv_mul_b, sv_a);

			svst1(pq, part1_w, sv_final);
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
	/*
		w = [
			0  w0r = 0 + 4
			1  w0i = 1 + 5
			2  w1r = 2 - 7
			3  w1i = 3 + 6
			4  w2r = 0 - 4
			5  w2i = 1 - 5
			6  w3r = 2 + 7
			7  w3i = 3 - 6
			8  w4r = 8 + 12
			9  w4i = 9 + 13
			10 w5r = 10 - 15
			11 w5i = 11 + 14
			12 w6r = 8 - 12
			13 w6i = 9 - 13
			14 w7r = 10 + 15
			15 w7i = 11 - 14
		]	
	*/
	float part2_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11};
		uint32_t indices_b[16] = {4, 5, 7, 6, 4, 5, 7, 6, 12, 13, 15, 14, 12, 13, 15, 14};
		float mul_b[16] = {1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indices_a = svld1(pq, indices_a + i);
			sv_a = svld1_gather_index(pq, part1_w, sv_indices_a);

			svuint32_t sv_indices_b = svld1(pq, indices_b + i);
			sv_b = svld1_gather_index(pq, part1_w, sv_indices_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i);
			
			svfloat32_t sv_final = svmad_f32_m(pq, sv_b, sv_mul_b, sv_a);

			svst1(pq, part2_w, sv_final);
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
	/*
		w = [
			0  w0r = 0 + 8,
			1  w0i = 1 + 9,
			2  w1r = 2 + 10,
			3  w1i = 3 + 11,
			4  w2r = 4 - 12,
			5  w2i = 5 + 13,
			6  w3r = 6 - 14,
			7  w3i = 7 + 15,
			8  w4r = 0 - 8,
			9  w4i = 1 - 9,
			10 w5r = 2 - 10,
			11 w5i = 3 - 11,
			12 w6r = 4 + 12,
			13 w6i = 5 - 13,
			14 w7r = 6 + 14,
			15 w7i = 7 - 15,
		]	
	*/
	float part3_w[16] = {0.0};
	{
		uint32_t indices_a[16] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
		uint32_t indices_b[16] = {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15};
		float mul_b[16] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

		const float scale = 0.125f;
		svfloat32_t sv_scale = svdup_f32(scale);

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indices_a = svld1(pq, indices_a + i);
			sv_a = svld1_gather_index(pq, part2_w, sv_indices_a);

			svuint32_t sv_indices_b = svld1(pq, indices_b + i);
			sv_b = svld1_gather_index(pq, part2_w, sv_indices_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i);

			svfloat32_t sv_final = svmad_f32_m(pq, sv_b, sv_mul_b, sv_a);
			sv_final = svmul_f32_m(pq, sv_final, sv_scale);

			svst1(pq, part3_w, sv_final);
		}
	}

	// Store outputs
	{
		svint32_t offsets = svindex_s32(0, stride_t * sizeof(float));
		uint64_t vector_len = svlen(offsets);

		for(uint32_t  i=0; i < 8; i+=vector_len){
			svbool_t pq = svwhilelt_b32_s32(i, 8); 
			svfloat32_t w_low = svld1(pq, part3_w + i);
			svfloat32_t w_high = svld1(pq, part3_w + 8 + i);
			svst1_scatter_offset(pq, t_lo, offsets, w_low);
			svst1_scatter_offset(pq, t_hi, offsets, w_high);
			t_lo += stride_t * vector_len;
			t_hi += stride_t * vector_len;
		}
	}
}