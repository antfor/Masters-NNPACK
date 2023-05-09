#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>
#include <sve/fft/complex-aos.h>

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
		cmulc_twiddle(&pg, &new_b, &twiddle, &new_bt);
		suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

		// stage2
		butterfly(&pg, &a, &b, &new_a, &new_b);

		// store
		svst1_scatter_offset(pg, f + i * 2 + 0, ind_store, new_a);
		svst1_scatter_offset(pg, f + i * 2 + 4, ind_store, new_b);
	}
}

static inline void stuff_for_fft8x8_sve(
	const float w[restrict static 1],
	uint32_t column_count,
	float f[restrict static 1],
	size_t stride_f)
{
	svbool_t pg, pg_real_active, pg_imag_active;
	svfloat32_t sv_w0, sv_w1, sv_w2, sv_w3;
	svfloat32_t W0, W1, W2, W3;
	svfloat32_t G1, H1, H_MP;

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_SIZE = 4;
	const uint32_t BLOCK_SIZEx2 = 16;

	const uint64_t numVals = svcntw() / 2;
	
	const svbool_t real_active = svdupq_b32(1, 0, 1, 0);
	const svbool_t imag_active = svdupq_b32(0, 1, 0, 1);
	
	const svfloat32_t half = svdup_f32(0.5f);
	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);
	const svfloat32_t neg_sqrt2_over_4 = svdup_f32(-SQRT2_OVER_4);
	const svfloat32_t zero = svdup_f32(0.0f);
	const uint32_t to_byte = sizeof(float);
	const svuint32_t ind_load = index2(to_byte * 0 ,to_byte * 1 , to_byte * 8);
	const svuint32_t ind_store = index2(to_byte * 0 ,to_byte * 8 , to_byte * 1);

	for (uint32_t column = 0; column < column_count; column += numVals)
	{
		pg = svwhilelt_b32(column * 2, column_count * 2);
		pg_real_active = svmov_z(pg, real_active);
		pg_imag_active = svmov_z(pg, imag_active);

		sv_w0 = svld1_gather_offset(pg, w + 0 + column * BLOCK_SIZE, ind_load);
		sv_w1 = svld1_gather_offset(pg, w + 2 + column * BLOCK_SIZE, ind_load);
		sv_w2 = svld1_gather_offset(pg, w + 4 + column * BLOCK_SIZE, ind_load);
		sv_w3 = svld1_gather_offset(pg, w + 6 + column * BLOCK_SIZE, ind_load);

		W0 = svcadd_m(pg, sv_w0, sv_w0, 270);
		W0 = svmul_m(pg, W0, to_conjugate);

		W2 = svmul_m(pg, sv_w2, to_conjugate);


		W1 = sv_w1;
		W3 = sv_w3;

		G1 = svadd_f32_m(pg_real_active, W1, W3); 
		G1 = svsub_f32_m(pg_imag_active, G1, W3); // G1 = iw1 + iw3 because merge
		G1 = svmul_m(pg, G1, half);

		H1 = svsub_f32_m(pg_real_active, W3, W1); 
		H1 = svadd_f32_m(pg_imag_active, H1, W1); 
		H1 = svcadd_m(pg, zero, H1, 270);
		H1 = svmul_m(pg, H1, to_conjugate);

		H_MP = svcadd_m(pg, H1, H1, 90);
		H_MP = svmul_m(pg, H_MP, neg_sqrt2_over_4);


		W1 = svcadd_m(pg, G1, H_MP, 90);
		W3 = svcadd_m(pg, G1, H_MP, 270);
		W3 = svmul_m(pg, W3, to_conjugate); 

		//store
		//svst1(pg, f + 0 * stride_f + column * 2, W0);
		//svst1(pg, f + 2 * stride_f + column * 2, W1);
		//svst1(pg, f + 4 * stride_f + column * 2, W2);
		//svst1(pg, f + 6 * stride_f + column * 2, W3);

		//todo use svst1
		svst1_scatter_offset(pg, f + 0 * stride_f + column, ind_store, W0);
		svst1_scatter_offset(pg, f + 2 * stride_f + column, ind_store, W1);
		svst1_scatter_offset(pg, f + 4 * stride_f + column, ind_store, W2);
		svst1_scatter_offset(pg, f + 6 * stride_f + column, ind_store, W3);

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

	stuff_for_fft8x8_sve(w, column_count, f, stride_f);
}


static inline void stuff_for_ifft8x8(float block[restrict static 1], uint32_t column_count){

	svbool_t pg, pg_real_active, pg_imag_active;
	svfloat32_t sv_w0, sv_w1, sv_w2, sv_w3;
	svfloat32_t W0, W1, W2, W3;
	svfloat32_t G1, H1, H_MP;

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_SIZE = 4;
	const uint32_t BLOCK_SIZEx2 = 16;

	const uint64_t numVals = svcntw() / 2;
	
	const svbool_t real_active = svdupq_b32(1, 0, 1, 0);
	const svbool_t imag_active = svdupq_b32(0, 1, 0, 1);
	
	const svfloat32_t half = svdup_f32(0.5f);
	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);
	const svfloat32_t sqrt2_over_2 = svdup_f32(SQRT2_OVER_2);

	for (uint32_t column = 0; column < column_count; column += numVals)
	{
		pg = svwhilelt_b32(column * 2, column_count * 2);
		pg_real_active = svmov_z(pg, real_active);
		pg_imag_active = svmov_z(pg, imag_active);

		sv_w0 = svld1(pg, block + column * 2 + BLOCK_SIZEx2 * 0);
		sv_w1 = svld1(pg, block + column * 2 + BLOCK_SIZEx2 * 1);
		sv_w2 = svld1(pg, block + column * 2 + BLOCK_SIZEx2 * 2);
		sv_w3 = svld1(pg, block + column * 2 + BLOCK_SIZEx2 * 3);

		W0 = svmul_m(pg, sv_w0, half);
		W0 = svcadd_m(pg, W0, W0, 270);
		W0 = svmul_m(pg, W0, to_conjugate);

		W2 = svmul_m(pg, sv_w2, to_conjugate);

		W1 = svmul_m(pg, sv_w1, half);
		W3 = svmul_m(pg, sv_w3, half);


		G1 = svadd_f32_m(pg_real_active, W1, W3); 
		G1 = svsub_f32_m(pg_imag_active, G1, W3); // G1 = iw1 + iw3 because merge

		H1 = svsub_f32_m(pg_real_active, W1, W3); 
		H1 = svadd_f32_m(pg_imag_active, H1, W3); 

		H_MP = svcadd_m(pg, H1, H1, 90);
		H_MP = svmul_m(pg, H_MP, sqrt2_over_2);

		W1 = svcadd_m(pg, G1, H_MP, 90);
		W3 = svcadd_m(pg, G1, H_MP, 270);
		W3 = svmul_m(pg, W3, to_conjugate); 

		//store
		svst1(pg, block + BLOCK_SIZEx2 * 0 + column * 2, W0);
		svst1(pg, block + BLOCK_SIZEx2 * 1 + column * 2, W1);
		svst1(pg, block + BLOCK_SIZEx2 * 2 + column * 2, W2);
		svst1(pg, block + BLOCK_SIZEx2 * 3 + column * 2, W3);

	}
}

static inline void ifft4xNc(	
	float block[restrict static 1],
	uint32_t column_count)
{

	const uint32_t BLOCK_SIZE = 8;
	const uint32_t HALF_BLOCK_SIZE = 4;
	const uint32_t HALF_BLOCK_LENGTH = 32;

	const svfloat32_t scaled_twiddle = svdupq_f32(0.25f * COS_0PI_OVER_2, 0.25f * SIN_0PI_OVER_2, 0.25f * COS_1PI_OVER_2, 0.25f * SIN_1PI_OVER_2);

	svbool_t pg;
	svfloat32_t a, b, new_a, new_b, new_bt;

	const float to_byte = sizeof(float); 
	const svuint32_t offsets = index4(to_byte * 0, to_byte * 1, to_byte * 16, to_byte * 17, to_byte * 2);
	const svfloat32_t scale = svdup_f32(0.25f);
	const uint64_t numVals = svcntw() / 4;

	const svuint32_t ind_zip = index4(0, 2, 1, 3, 4);
	const svuint32_t ind_low = index2(0, 1, 4);
	const svuint32_t ind_high = index2(2, 3, 4);

	for(uint32_t column = 0; column < column_count; column += numVals){

		pg = svwhilelt_b32_s32(column * 4, column_count * 4);

		// load
		a = svld1_gather_offset(pg, block + column * 2 + 0, offsets);
		b = svld1_gather_offset(pg, block + column * 2 + HALF_BLOCK_LENGTH, offsets);

		// stage1
		butterfly(&pg, &a, &b, &new_a, &new_b);
		cmulc_twiddle(&pg, &new_b, &scaled_twiddle, &new_bt);
		new_a = svmul_m(pg, new_a, scale);
		suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

		// stage2
		butterfly(&pg, &a, &b, &new_a, &new_b);

		// store
		svst1_scatter_offset(pg, block + column * 2 + 0 , offsets, new_a);
		svst1_scatter_offset(pg, block + column * 2 + HALF_BLOCK_LENGTH, offsets, new_b);

	}
}


static inline void sve_ifft8x8_real(
	float block[restrict static 1],
	uint32_t column_count)
{
	zip_rows_8(block); // todo try and remove
	stuff_for_ifft8x8(block, column_count);
	ifft4xNc(block, column_count);
}

//--16x16------------------------------------------------------------


static inline void complex_to_real_NxNc_512(const float w[restrict static 1], float f[restrict static 1], uint32_t column_offset, uint32_t column_count, int N, svfloat32_t twiddle_i){

	const uint32_t BLOCK_SIZE = N/2;

	const int offset = 2; //skip w0
    const int HL = column_count * 8; //half length - vector b starts here 
    const uint64_t numVals = svcntw()/8;

	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);
	const svfloat32_t i = svdupq_f32(0.0f, 1.0f, 0.0f, 1.0f);

	svbool_t pg;
	const svbool_t all = svptrue_b32();
	svfloat32_t xr, xN_r,x, xe, xo, xot;

	const svuint32_t indr =   indexA(all, (uint32_t []){2,3,4,5,6,7,0+HL,1+HL}, 8, 8);
	const svuint32_t indN_r = indexA(all, (uint32_t []){6+HL,7+HL,4+HL,5+HL,2+HL,3+HL,0+HL,1+HL}, 8, 8);

	const svuint32_t ind_store_top = indexN(all, offset * 16, 16, 1, 8);
	const svuint32_t ind_store_bot = indexA(all, (uint32_t []){6*16,7*16,4*16,5*16,2*16,3*16,0*16,1*16}, 8, 1);


	for(int column = 0; column < column_count; column+=numVals){

		pg = svwhilelt_b32_s32(column * 8, column_count * 8);

		//load 
		xr  =  svld1_gather_index(pg, w + column * 8 , indr);
		xN_r = svld1_gather_index(pg, w + column * 8 , indN_r);

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

		svst1_scatter_index(pg, f + column_offset + column + 8 * 16, ind_store_bot, x);
 
	}
}


static inline void sve_fft16_real(
	const float t0[restrict static 1],
	const float t8[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	uint32_t column_offset, uint32_t column_count,
	float f[restrict static 1])
{

	float w[16 * column_count]; //todo make in place

	fft8xNr(t0, t8, stride_t, row_offset, row_count, column_offset, column_count, w);
    
	{
		const svfloat32_t twiddle_top_i = svzip1(svdupq_f32(-SIN_1PI_OVER_8, -SIN_2PI_OVER_8, -SIN_3PI_OVER_8, -SIN_4PI_OVER_8),
									             svdupq_f32(COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8));

		complex_to_real_NxNc_512(w, f, column_offset, column_count, 16, twiddle_top_i);
	}

	fftN_to_fft2N(w,8,f,16,column_offset,column_count);

}

static inline void icomplex_to_real_NxNc_512(const float f[restrict static 1], float t[restrict static 1], uint32_t column_count, int N, svfloat32_t twiddle_i){

	const uint32_t BLOCK_SIZE = N/2;
	const uint32_t stride = N;

    const uint64_t numVals = svcntw()/8;

	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);

	svbool_t pg;
	const svbool_t all = svptrue_b32();
	svfloat32_t xr, xN_r,x, xe, xo, xot;

	//todo make indexN
	const svuint32_t indr =   indexA(all, (uint32_t []){16*1, 16*1 + 128,16*2, 16*2 + 128,16*3, 16*3 + 128,16*4, 16*4 + 128}, 8, 1);
	const svuint32_t indN_r =   indexA(all, (uint32_t []){16*7, 16*7 + 128,16*6, 16*6 + 128,16*5, 16*5 + 128,16*4, 16*4 + 128}, 8, 1);


	const svuint32_t ind_store_top = indexN(all, 2, 1, 16, 8);
	//todo indexN?
	const svuint32_t ind_store_bot = indexA(all, (uint32_t []){6,7,4,5,2,3,0,1}, 8, 16);

	for(int column = 0; column < column_count; column+=numVals){

		pg = svwhilelt_b32_s32(column * BLOCK_SIZE, column_count * BLOCK_SIZE);

		//load 
		xr  =  svld1_gather_index(pg, f + column, indr);
		xN_r = svld1_gather_index(pg, f + column, indN_r);
		xN_r = svmul_m(pg, xN_r, to_conjugate);

		xe = svadd_m(pg, xr, xN_r);
		xe = svmul_m(pg, xe, 0.5f);

		xo = svsub_m(pg, xr, xN_r);
		xo = svmul_m(pg, xo, 0.5f);

		cmul_twiddle(&pg, &xo, &twiddle_i, &xot);


		x = svadd_m(pg, xe, xot);
		svst1_scatter_index(pg, t + column * stride, ind_store_top, x);

		x = svsub_m(pg, xe, xot);
		x = svmul_m(pg, x, to_conjugate);
		svst1_scatter_index(pg, t + column * stride + BLOCK_SIZE, ind_store_bot, x);
	}
}


static inline void sve_ifft16x16_real(float block[restrict static 256], size_t column_count){
	
	float w[16 * column_count]; //todo make in place

	{
		const svfloat32_t twiddle_top_i = svzip1(svdupq_f32(-SIN_1PI_OVER_8, -SIN_2PI_OVER_8, -SIN_3PI_OVER_8, -SIN_4PI_OVER_8),
												svdupq_f32(COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8));

		icomplex_to_real_NxNc_512(block, w, column_count, 16, twiddle_top_i);
	}

	ifftN_to_ifft2N(block, 128, w, 16, column_count);

	sve_ifft8xNr(w, block, column_count);

}