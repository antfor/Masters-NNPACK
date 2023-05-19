#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/aos.h>
#include <arm_sve.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>
#include <sve/fft/complex-aos.h>
#include <sve/fft/complex-channel-aos.h>


//--fftN to fft2N-------------------------------------------------------------

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

inline static void fftN_to_fft2N_channel(
	const float w[restrict static 1],
	size_t stride_c,

	float f[restrict static 1],
    size_t stride_r,
    const uint32_t column_offset,
	const uint32_t column_count,
	int channels, int N)
{

    svfloat32_t a, b, new_a, new_b;
    svbool_t pg;
    uint32_t numVals = svcntw();

	const int stride_w = column_count * N;
	const int stride_f = N * N;

	const svbool_t all = svptrue_b32(); 
	const svuint32_t ind_load =  svindex_u32(0, stride_w); 
	const svuint32_t ind_store =  svindex_u32(0, stride_f);


    for(int column =0; column < column_count; column += 1 ){

		for(int channel=0; channel < channels; channel+= numVals){
			//pg = svwhilelt_b32_s32(column, column + 1);
			pg = svwhilelt_b32_s32(channel, channels);

			//load
			a = svld1_gather_index(pg, w + channel * stride_w + column * stride_c + 0, ind_load);
			b = svld1_gather_index(pg, w + channel * stride_w + column * stride_c + 1, ind_load);
		
			//stage 1
			butterfly(&pg, &a, &b, &new_a, &new_b);

			//store
			svst1_scatter_index(pg, f + channel * stride_f + column_offset + column + 0,        ind_store, new_a);
			svst1_scatter_index(pg, f + channel * stride_f + column_offset + column + stride_r, ind_store, new_b);
		}

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
//--twiddle---------------------------------------------------------------------

static inline svfloat32_t get_twiddle_i_top(int BLOCK_SIZE){

	switch(BLOCK_SIZE){
	case 4:
		return svdupq_f32(-SIN_1PI_OVER_4, COS_1PI_OVER_4, -SIN_2PI_OVER_4, COS_2PI_OVER_4);
	
	case 8:
		return svzip1(svdupq_f32(-SIN_1PI_OVER_8, -SIN_2PI_OVER_8, -SIN_3PI_OVER_8, -SIN_4PI_OVER_8),
					  svdupq_f32(COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8));
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

		//note complex_to_real_NxNc_channel depend on the jump value 
		switch(BLOCK_SIZE){
			case 4:
				*indr =   indexA(all, (uint32_t []){2,3,0+HL,1+HL}, BLOCK_SIZE, BLOCK_SIZE);
				*indN_r = indexA(all, (uint32_t []){2+HL,3+HL,0+HL,1+HL}, BLOCK_SIZE, BLOCK_SIZE);

				*ind_store_top = indexN(all, offset * N, N, 1, BLOCK_SIZE);
				*ind_store_bot = indexA(all, (uint32_t []){2*N,3*N,0*N,1*N}, BLOCK_SIZE, 1);
				break;

			case 8:
				*indr =   indexA(all, (uint32_t []){2,3,4,5,6,7,0+HL,1+HL}, 8, 8);
				*indN_r = indexA(all, (uint32_t []){6+HL,7+HL,4+HL,5+HL,2+HL,3+HL,0+HL,1+HL}, 8, 8);

				*ind_store_top = indexN(all, offset * 16, 16, 1, 8);
				*ind_store_bot = indexA(all, (uint32_t []){6*16,7*16,4*16,5*16,2*16,3*16,0*16,1*16}, 8, 1);
				break;
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

	svbool_t pg;
	svfloat32_t xr, xN_r,x, xe, xo, xot;
	svuint32_t indr, indN_r, ind_store_top, ind_store_bot;
	ctr_get_index(BLOCK_SIZE, column_count, N, &indr, &indN_r, &ind_store_top, &ind_store_bot);
	svfloat32_t twiddle_i = get_twiddle_i_top(BLOCK_SIZE);

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


static inline void complex_to_real_NxNc_channel(
	const float w[restrict static 1], 
	float f[restrict static 1],
	uint32_t column_offset, uint32_t column_count, 
	int N, int channels){

	const uint32_t BLOCK_SIZE = N/2;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;

	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);

	svbool_t pg;
	svfloat32_t xr, xN_r,x, xe, xo, xot;
	svuint32_t indr, indN_r, ind_store_top, ind_store_bot;
	ctr_get_index(BLOCK_SIZE, column_count, N, &indr, &indN_r, &ind_store_top, &ind_store_bot);
	svfloat32_t twiddle_i = get_twiddle_i_top(BLOCK_SIZE);


	const svbool_t all = svptrue_b32();
	const int w_stride = N * column_count;
	const int f_stride = N*N;
	//jump value can be diffret for riscV?
	const svuint32_t w_offset = repeatN(all, 0, w_stride - BLOCK_SIZE, BLOCK_SIZE);
	const svuint32_t f_offset = repeatN(all, 0, f_stride - 1, BLOCK_SIZE);

	indr = svadd_m(all, indr, w_offset);
	indN_r = svadd_m(all, indN_r, w_offset);
	ind_store_top = svadd_m(all, ind_store_top, f_offset);
	ind_store_bot = svadd_m(all, ind_store_bot, f_offset);

	for(int column = 0; column < column_count; column+=1){

		//numVals = 1;
		for(uint32_t channel = 0; channel < channels; channel += numVals){
		
			//pg = svwhilelt_b32_s32(column * BLOCK_SIZE, (column + 1) * BLOCK_SIZE);
			pg = svwhilelt_b32_s32(channel * BLOCK_SIZE , channels * BLOCK_SIZE);

			//load 
			xr  =  svld1_gather_index(pg, w + channel * w_stride + column * BLOCK_SIZE , indr);
			xN_r = svld1_gather_index(pg, w + channel * w_stride + column * BLOCK_SIZE , indN_r);
			xN_r = svmul_m(pg, xN_r, to_conjugate);

			xe = svadd_m(pg, xr, xN_r);
			xe = svmul_m(pg, xe, 0.5f);

			xo = svsub_m(pg, xr, xN_r);
			xo = svmul_m(pg, xo, 0.5f);

			cmulc_twiddle(&pg, &xo, &twiddle_i, &xot);

			x = svadd_m(pg, xe, xot);
			svst1_scatter_index(pg, f + channel * f_stride + column_offset + column + 0, ind_store_top, x);

			x = svsub_m(pg, xe, xot);
			x = svmul_m(pg, x, to_conjugate);
			svst1_scatter_index(pg, f + channel * f_stride + column_offset + column + BLOCK_SIZE * N, ind_store_bot, x);
	
		}

	}
}

//--real to complex-------------------------------------------------------------

static inline void rtc_get_index(
	int BLOCK_SIZE, uint32_t column_count, int N,
	svuint32_t *indr,
	svuint32_t *indN_r,
	svuint32_t *ind_store_top,
	svuint32_t *ind_store_bot){

	const svbool_t all = svptrue_b32();
	const int offset = 2; //skip w0
    const int HL = column_count * BLOCK_SIZE;
		switch(BLOCK_SIZE){
			case 4:
				*indr =   indexA(all, (uint32_t []){N*1, N*1 + 32,N*2, N*2 + 32}, 4, 1);
				*indN_r = indexA(all, (uint32_t []){N*3, N*3 + 32,N*2, N*2 + 32}, 4, 1);
				
				*ind_store_top = indexN(all, 2, 1, N, 4);
				*ind_store_bot = indexA(all, (uint32_t []){2,3,0,1}, 4, N);
				break;

			case 8:
				//todo make indexN
				*indr =   indexA(all, (uint32_t []){N*1, N*1 + 128,N*2, N*2 + 128,N*3, N*3 + 128,N*4, N*4 + 128}, 8, 1);
				*indN_r = indexA(all, (uint32_t []){N*7, N*7 + 128,N*6, N*6 + 128,N*5, N*5 + 128,N*4, N*4 + 128}, 8, 1);
				
				*ind_store_top = indexN(all, 2, 1, N, 8);
				*ind_store_bot = indexA(all, (uint32_t []){6,7,4,5,2,3,0,1}, 8, N);
				break;
		}	
}

static inline void real_to_complex_NxNc(
	const float f[restrict static 1], 
	float t[restrict static 1],
	 uint32_t column_count, int N){

	const uint32_t BLOCK_SIZE = N/2;
	const uint32_t stride = N;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;

	const svfloat32_t to_conjugate = svdupq_f32(1.0f,-1.0f, 1.0f,-1.0f);

	svbool_t pg;
	svfloat32_t xr, xN_r,x, xe, xo, xot;
	svuint32_t indr, indN_r, ind_store_top, ind_store_bot;
	rtc_get_index(BLOCK_SIZE, column_count, N, &indr, &indN_r, &ind_store_top, &ind_store_bot);
	svfloat32_t twiddle_i = get_twiddle_i_top(BLOCK_SIZE);

	for(int column = 0; column < column_count; column+=numVals){

		pg = svwhilelt_b32_s32(column * BLOCK_SIZE, column_count * BLOCK_SIZE);

		//load 
		xr  =  svld1_gather_index(pg, f + column, indr);
		xN_r = svld1_gather_index(pg, f + column, indN_r);
		xN_r = svmul_m(pg, xN_r, to_conjugate);

		//printf("in xr, xN_r\n");
		//svprint_f(pg, xr, 16);
		//svprint_f(pg, xN_r, 16);

		xe = svadd_m(pg, xr, xN_r);
		xe = svmul_m(pg, xe, 0.5f);

		xo = svsub_m(pg, xr, xN_r);
		xo = svmul_m(pg, xo, 0.5f);

		cmul_twiddle(&pg, &xo, &twiddle_i, &xot);

		x = svadd_m(pg, xe, xot);
		svst1_scatter_index(pg, t + column * stride + 0, ind_store_top, x);

		x = svsub_m(pg, xe, xot);
		x = svmul_m(pg, x, to_conjugate);
		svst1_scatter_index(pg, t + column * stride + BLOCK_SIZE, ind_store_bot, x);
 
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

	complex_to_real_NxNc(w, f, column_offset, column_count, 8);
	fftN_to_fft2N(w,4,f,8,column_offset,column_count);
}


static inline void sve_ifft8x8_real(
	float block[restrict static 1],
	uint32_t column_count)
{
	float w[8 * column_count];

	real_to_complex_NxNc(block, w, column_count, 8);
	ifftN_to_ifft2N(block, 32, w, 8, column_count);

	ifft4xNc(w, block, column_count);
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

static inline void sve_fft16x16_real_kernel(
	float t0[restrict static 1],
	float t8[restrict static 1],
	size_t stride_t,
	uint32_t row_count,
    uint32_t column_count,
	float f[restrict static 1],
	int channels)
{
	//todo make in place
	float w[16 * column_count * channels]; 

	fft8xNr_channel(t0, t8, stride_t, 0, row_count, 0, column_count, w, channels);

	complex_to_real_NxNc_channel(w, f, 0, column_count, 16, channels);

	fftN_to_fft2N_channel(w,8,f,16,0,column_count, channels, 16);
}

static inline void sve_ifft16x16_real(float block[restrict static 256], size_t column_count){
	
	float w[16 * column_count]; //todo make in place

	real_to_complex_NxNc(block, w, column_count, 16);
	ifftN_to_ifft2N(block, 128, w, 16, column_count);

	sve_ifft8xNr(w, block, column_count);

}