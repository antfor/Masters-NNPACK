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

static inline void sve_fft8_aos(
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
		float old_w[16] = {w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			sv_a = svld1_gather_index(pq, old_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			sv_b = svld1_gather_index(pq, old_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);
			
			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svst1(pq, part1_w, sv_added);
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

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			sv_a = svld1_gather_index(pq, part1_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			sv_b = svld1_gather_index(pq, part1_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);
			
			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svst1(pq, part2_w, sv_added);
		}
	}

	*f0r = part2_w[0];
	*f0i = part2_w[1];
	*f1r = part2_w[2];
	*f1i = part2_w[3];
	*f2r = part2_w[4];
	*f2i = part2_w[5];
	*f3r = part2_w[6];
	*f3i = part2_w[7];
	*f4r = part2_w[8];
	*f4i = part2_w[9];
	*f5r = part2_w[10];
	*f5i = part2_w[11];
	*f6r = part2_w[12];
	*f6i = part2_w[13];
	*f7r = part2_w[14];
	*f7i = part2_w[15];
}

static inline void sve_ifft8_aos(
	float w0r, float w0i, float w1r, float w1i, float w2r, float w2i, float w3r, float w3i,
	float w4r, float w4i, float w5r, float w5i, float w6r, float w6i, float w7r, float w7i,
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
		float old_w[16] = {w0r, w0i, w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i, w5r, w5i, w6r, w6i, w7r, w7i};

		svbool_t pq;
		svfloat32_t sv_a, sv_b;

		uint64_t vector_len = svlen_f32(sv_a);
		for(uint32_t  i=0; i < 16; i+=vector_len){
			pq = svwhilelt_b32_s32(i, 16); 

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			sv_a = svld1_gather_index(pq, old_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			sv_b = svld1_gather_index(pq, old_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);
			
			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svst1(pq, part1_w, sv_added);
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

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			sv_a = svld1_gather_index(pq, part1_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			sv_b = svld1_gather_index(pq, part1_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);
			
			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svst1(pq, part2_w, sv_added);
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

			svuint32_t sv_indexes_a = svld1(pq, indices_a + i * vector_len);
			sv_a = svld1_gather_index(pq, part2_w, sv_indexes_a);

			svuint32_t sv_indexes_b = svld1(pq, indices_b + i * vector_len);
			sv_b = svld1_gather_index(pq, part2_w, sv_indexes_b);

			svfloat32_t sv_mul_b = svld1(pq, mul_b + i * vector_len);
			sv_b = svmul_f32_m(pq, sv_b, sv_mul_b);
			
			svfloat32_t sv_added = svadd_f32_m(pq, sv_a, sv_b);
			svfloat32_t sv_scaled = svmul_f32_m(pq, sv_added, sv_scale);

			svst1(pq, part3_w, sv_scaled);
		}
	}

	// Store outputs
	{
		svint32_t offsets = svindex_s32(0, stride_t * sizeof(float));
		uint64_t vector_len = svlen(offsets);

		for(uint32_t  i=0; i < 8; i+=vector_len){
			svbool_t pq = svwhilelt_b32_s32(i, 8); 
			svfloat32_t w_low = svld1(pq, part3_w + i * vector_len);
			svfloat32_t w_high = svld1(pq, part3_w + 8 + i * vector_len);
			svst1_scatter_offset(pq, t_lo, offsets, w_low);
			svst1_scatter_offset(pq, t_hi, offsets, w_high);
			t_lo += stride_t * vector_len;
			t_hi += stride_t * vector_len;
		}
	}
}

/* Opimization ideas:
1. Use multiply and add instruction instead of mul and then add
2. In some cases can avoid gather / scatter
3. Possibly use negation instead of multiply in some cases?
*/