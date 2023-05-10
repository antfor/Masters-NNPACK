#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>
#include <sve/fft/complex-aos.h>


//--real to complex-------------------------------------------------------------

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

}


inline static void ifftN_to_ifft2N(
	const float w[restrict static 1],
	size_t offset_to_b,

	float f[restrict static 1],
    size_t stride_f,
	const uint32_t column_count)
{

    svfloat32_t a, b, new_a, new_b;
    svbool_t pg;
    uint32_t numVals = svcntw();

    const svuint32_t ind_store = svindex_u32(0, stride_f);

    for(int column =0; column < column_count; column += numVals ){
        pg = svwhilelt_b32_s32(column, column_count);

        //load
        a = svld1(pg, w + column + 0);
        b = svld1(pg, w + column + offset_to_b);
    
        //stage 1
        butterfly(&pg, &a, &b, &new_a, &new_b);

        new_a = svmul_m(pg, new_a, 0.5f);
        new_b = svmul_m(pg, new_b, 0.5f);

        //store
         svst1_scatter_index(pg, f + column * stride_f + 0, ind_store, new_a);
         svst1_scatter_index(pg, f + column * stride_f + 1, ind_store, new_b);
    }

}

//--complex to real-------------------------------------------------------------
 
static inline void ctr_get_index(
	int BLOCK_SIZE, uint32_t column_count, int N,
	svuint32_t *indr,
	svuint32_t *indN_r,
	svuint32_t *ind_store_top,
	svuint32_t *ind_store_bot){

	const svbool_t all = svptrue_b32();
	const int offset = 2; //skip w0
    const int HL = column_count * BLOCK_SIZE;
		switch(N){
			case 8:
				*indr =   indexA(all, (uint32_t []){2,3,0+HL,1+HL}, BLOCK_SIZE, BLOCK_SIZE);
				*indN_r = indexA(all, (uint32_t []){2+HL,3+HL,0+HL,1+HL}, BLOCK_SIZE, BLOCK_SIZE);

				*ind_store_top = indexN(all, offset * N, N, 1, BLOCK_SIZE);
				*ind_store_bot = indexA(all, (uint32_t []){2*N,3*N,0*N,1*N}, BLOCK_SIZE, 1);
				break;

			case 16:
				*indr =   indexA(all, (uint32_t []){2,3,4,5,6,7,0+HL,1+HL}, 8, 8);
				*indN_r = indexA(all, (uint32_t []){6+HL,7+HL,4+HL,5+HL,2+HL,3+HL,0+HL,1+HL}, 8, 8);

				*ind_store_top = indexN(all, offset * 16, 16, 1, 8);
				*ind_store_bot = indexA(all, (uint32_t []){6*16,7*16,4*16,5*16,2*16,3*16,0*16,1*16}, 8, 1);
				break;
		}	
}

static inline svfloat32_t ctr_get_twiddle_i_top(int BLOCK_SIZE){

	switch(BLOCK_SIZE){
	case 4:
		return svdupq_f32(-SIN_1PI_OVER_4, COS_1PI_OVER_4, -SIN_2PI_OVER_4, COS_2PI_OVER_4);
	
	case 8:
		return svzip1(svdupq_f32(-SIN_1PI_OVER_8, -SIN_2PI_OVER_8, -SIN_3PI_OVER_8, -SIN_4PI_OVER_8),
					  svdupq_f32(COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8));
	}

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

//--8x8--------------------------------------------------------------


static inline void sve_fft8xN_real(
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

	//printf("w\n");
	//fprint_array_f(w, 8*column_count, 8);

	
	complex_to_real_NxNc(w, f, column_offset, column_count, 8);
	fftN_to_fft2N(w,4,f,8,column_offset,column_count);
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

static inline void sve_fft16x16_real(
	const float t0[restrict static 1],
	const float t8[restrict static 1],
	size_t stride_t,
	uint32_t row_offset, uint32_t row_count,
	uint32_t column_offset, uint32_t column_count,
	float f[restrict static 1])
{
	float w[16 * column_count]; //todo make in place

	fft8xNr(t0, t8, stride_t, row_offset, row_count, column_offset, column_count, w);

	complex_to_real_NxNc(w, f, column_offset, column_count, 16);	
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