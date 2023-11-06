#pragma once

#include <nnpack/fft-constants.h>
#include <riscvv/fft/aos.h>
#include <riscvv/fft/fft-util.h>
#include <riscvv/fft/complex.h>
#include <nnpack/hwinfo.h>
#include <riscvv/fft/complex-aos.h>

#include <riscvv/fft/rv-printf.h>

//--fftN to fft2N-------------------------------------------------------------

inline static void fftN_to_fft2N(
	const float w[restrict static 1],
	size_t stride_w,

	float f[restrict static 1],
    size_t stride_f,
    const uint32_t column_offset,
	const uint32_t column_count)
{

	__epi_2xf32 a, new_a;
	__epi_2xf32 b, new_b;

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = simd_width;

    int pg = imin(simd_width, column_count);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xi32 ind_load = rvindex_adress(0, stride_w, gvl);

	for(int column =0; column < column_count; column += numVals ){

		int pg = imin(simd_width, column_count - column);
		long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);
       	
		//load
		a = __builtin_epi_vload_indexed_2xf32(w + column * stride_w + 0, ind_load, gvl);
		b = __builtin_epi_vload_indexed_2xf32(w + column * stride_w + 1, ind_load, gvl);

		//stage 1
		new_a = butterfly_add(a, b, gvl);
		new_b = butterfly_sub(a, b, gvl);

		//store TODO INDEX?
		__builtin_epi_vstore_2xf32(f + column_offset + column + 0, new_a, gvl);
		__builtin_epi_vstore_2xf32(f + column_offset + column + stride_f, new_b, gvl);

	}
}


inline static void ifftN_to_ifft2N(
	const float w[restrict static 1],
	size_t offset_to_b,

	float f[restrict static 1],
    size_t stride_f,
	const uint32_t column_count)
{
	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = simd_width;

	int pg = imin(simd_width, column_count);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 a, new_a;
	__epi_2xf32 b, new_b;
	__epi_2xf32 half = __builtin_epi_vfmv_v_f_2xf32(0.5f, gvl);

	__epi_2xi32 ind_load = rvindex_adress(0, stride_f, gvl);

	for(int column =0; column < column_count; column += numVals ){

		//load
		a =__builtin_epi_vload_2xf32(w + column + 0, gvl);
		b =__builtin_epi_vload_2xf32(w + column + offset_to_b, gvl);

		//stage 1
		new_a = butterfly_add(a, b, gvl);
		new_b = butterfly_sub(a, b, gvl);

		new_a = __builtin_epi_vfmul_2xf32(new_a, half, gvl);
		new_b = __builtin_epi_vfmul_2xf32(new_b, half, gvl);

		//store
		__builtin_epi_vstore_indexed_2xf32(f + column * stride_f + 0, new_a, ind_load, gvl);
		__builtin_epi_vstore_indexed_2xf32(f + column * stride_f + 1, new_b, ind_load, gvl);
	}

}

//--complex to real-------------------------------------------------------------
 

static inline void complex_to_real_NxNc(
	const float w[restrict static 1], 
	float f[restrict static 1],
	uint32_t column_offset, uint32_t column_count, 
	int N){
	
	const uint32_t BLOCK_SIZE = N/2;

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (2 * simd_width) / BLOCK_SIZE;

    int pg = imin(simd_width, column_count * BLOCK_SIZE /2);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 twiddle_i_r = get_twiddle_i_top_r(BLOCK_SIZE, gvl);
	__epi_2xf32 twiddle_i_i = get_twiddle_i_top_i(BLOCK_SIZE, gvl);

	__epi_2xf32 to_conjugate = __builtin_epi_vfmv_v_f_2xf32(-1.0f, gvl);
	__epi_2xf32 half = __builtin_epi_vfmv_v_f_2xf32(0.5f, gvl);

	__epi_2xf32 xr_r, xNr_r,x_r, xe_r, xo_r, xot_r;
	__epi_2xf32 xr_i, xNr_i,x_i, xe_i, xo_i, xot_i;
	
	__epi_2xi32 indr, indN_r, ind_store_top, ind_store_bot;

	indr = ctr_get_indr(BLOCK_SIZE, column_count, N, gvl);
	indN_r = ctr_get_indN_r(BLOCK_SIZE, column_count, N, gvl);
	ind_store_top = ctr_get_ind_store_top(BLOCK_SIZE, column_count, N, gvl);
	ind_store_bot = ctr_get_ind_store_bot(BLOCK_SIZE, column_count, N, gvl);


	for(int column = 0; column < column_count; column+=numVals){

		int pg = imin(simd_width, column_count * BLOCK_SIZE /2 - column * BLOCK_SIZE /2);
   		long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

		//load
		xr_r = __builtin_epi_vload_indexed_2xf32(w + column * BLOCK_SIZE + 0, indr, gvl);
		xr_i = __builtin_epi_vload_indexed_2xf32(w + column * BLOCK_SIZE + 1, indr, gvl);

		xNr_r =__builtin_epi_vload_indexed_2xf32(w + column * BLOCK_SIZE + 0, indN_r, gvl);
		xNr_i =__builtin_epi_vload_indexed_2xf32(w + column * BLOCK_SIZE + 1, indN_r, gvl);

		//to real
		xNr_i = __builtin_epi_vfmul_2xf32(xNr_i, to_conjugate, gvl);

		xe_r = __builtin_epi_vfadd_2xf32(xr_r, xNr_r, gvl);
		xe_i = __builtin_epi_vfadd_2xf32(xr_i, xNr_i, gvl);
		xe_r = __builtin_epi_vfmul_2xf32(xe_r, half, gvl);
		xe_i = __builtin_epi_vfmul_2xf32(xe_i, half, gvl);

		xo_r = __builtin_epi_vfsub_2xf32(xr_r, xNr_r, gvl);
		xo_i = __builtin_epi_vfsub_2xf32(xr_i, xNr_i, gvl);
		xo_r = __builtin_epi_vfmul_2xf32(xo_r, half, gvl);
		xo_i = __builtin_epi_vfmul_2xf32(xo_i, half, gvl);

		xot_r = mulc_twiddle_r(xo_r, xo_i, twiddle_i_r, twiddle_i_i, gvl);
		xot_i = mulc_twiddle_i(xo_r, xo_i, twiddle_i_r, twiddle_i_i, gvl);

		//store
		x_r = __builtin_epi_vfadd_2xf32(xe_r, xot_r, gvl);
		x_i = __builtin_epi_vfadd_2xf32(xe_i, xot_i, gvl);
		__builtin_epi_vstore_indexed_2xf32(f + column_offset + column + 0, x_r, ind_store_top, gvl);
		__builtin_epi_vstore_indexed_2xf32(f + column_offset + column + N, x_i, ind_store_top, gvl);

		//store
		x_r = __builtin_epi_vfsub_2xf32(xe_r, xot_r, gvl);
		x_i = __builtin_epi_vfsub_2xf32(xe_i, xot_i, gvl);
		x_i = __builtin_epi_vfmul_2xf32(x_i, to_conjugate, gvl);
		__builtin_epi_vstore_indexed_2xf32(f + column_offset + column + BLOCK_SIZE * N + 0, x_r, ind_store_bot ,gvl);
		__builtin_epi_vstore_indexed_2xf32(f + column_offset + column + BLOCK_SIZE * N + N, x_i, ind_store_bot ,gvl);

	}
}

//--real to complex-------------------------------------------------------------

static inline void real_to_complex_NxNc(
	const float f[restrict static 1], 
	float t[restrict static 1],
	 uint32_t column_count, int N){

	const uint32_t BLOCK_SIZE = N/2;
	const uint32_t stride = N;
	const int HALF_LENGTH = N * BLOCK_SIZE;

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, column_count * BLOCK_SIZE /2);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 twiddle_i_r = get_twiddle_i_top_r(BLOCK_SIZE, gvl);
	__epi_2xf32 twiddle_i_i = get_twiddle_i_top_i(BLOCK_SIZE, gvl);

	__epi_2xf32 to_conjugate = __builtin_epi_vfmv_v_f_2xf32(-1.0f, gvl);
	__epi_2xf32 half = __builtin_epi_vfmv_v_f_2xf32(0.5f, gvl);

	__epi_2xf32 xr_r, xNr_r,x_r, xe_r, xo_r, xot_r;
	__epi_2xf32 xr_i, xNr_i,x_i, xe_i, xo_i, xot_i;
	
	__epi_2xi32 indr, indN_r, ind_store_top, ind_store_bot;

	indr = rtc_get_indr(BLOCK_SIZE, column_count, N, gvl);
	indN_r = rtc_get_indN_r(BLOCK_SIZE, column_count, N, gvl);
	ind_store_top = rtc_get_ind_store_top(BLOCK_SIZE, column_count, N, gvl);
	ind_store_bot = rtc_get_ind_store_bot(BLOCK_SIZE, column_count, N, gvl);


	for(int column = 0; column < column_count; column+=numVals){

		int pg = imin(simd_width, column_count * BLOCK_SIZE /2 - column * BLOCK_SIZE /2);
   		long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

		//load
		xr_r = __builtin_epi_vload_indexed_2xf32(f + column + 0, indr, gvl);
		xr_i = __builtin_epi_vload_indexed_2xf32(f + column + HALF_LENGTH, indr, gvl);

		xNr_r =__builtin_epi_vload_indexed_2xf32(f + column + 0, indN_r, gvl);
		xNr_i =__builtin_epi_vload_indexed_2xf32(f + column + HALF_LENGTH, indN_r, gvl);
		
		xNr_i = __builtin_epi_vfmul_2xf32(xNr_i, to_conjugate, gvl);

		xe_r = __builtin_epi_vfadd_2xf32(xr_r, xNr_r, gvl);
		xe_i = __builtin_epi_vfadd_2xf32(xr_i, xNr_i, gvl);
		xe_r = __builtin_epi_vfmul_2xf32(xe_r, half, gvl);
		xe_i = __builtin_epi_vfmul_2xf32(xe_i, half, gvl);

		xo_r = __builtin_epi_vfsub_2xf32(xr_r, xNr_r, gvl);
		xo_i = __builtin_epi_vfsub_2xf32(xr_i, xNr_i, gvl);
		xo_r = __builtin_epi_vfmul_2xf32(xo_r, half, gvl);
		xo_i = __builtin_epi_vfmul_2xf32(xo_i, half, gvl);

		xot_r = mul_twiddle_r(xo_r, xo_i, twiddle_i_r, twiddle_i_i, gvl);
		xot_i = mul_twiddle_i(xo_r, xo_i, twiddle_i_r, twiddle_i_i, gvl);

		//store
		x_r = __builtin_epi_vfadd_2xf32(xe_r, xot_r, gvl);
		x_i = __builtin_epi_vfadd_2xf32(xe_i, xot_i, gvl);


		__builtin_epi_vstore_indexed_2xf32(t + column * stride + 0, x_r, ind_store_top, gvl);
		__builtin_epi_vstore_indexed_2xf32(t + column * stride + 1, x_i, ind_store_top, gvl);


		x_r = __builtin_epi_vfsub_2xf32(xe_r, xot_r, gvl);
		x_i = __builtin_epi_vfsub_2xf32(xe_i, xot_i, gvl);
		x_i = __builtin_epi_vfmul_2xf32(x_i, to_conjugate, gvl);

		__builtin_epi_vstore_indexed_2xf32(t + column * stride + BLOCK_SIZE + 0, x_r, ind_store_bot ,gvl);
		__builtin_epi_vstore_indexed_2xf32(t + column * stride + BLOCK_SIZE + 1, x_i, ind_store_bot ,gvl);
	}

}
//--8x8--------------------------------------------------------------

static inline void riscvv_fft8xN_real(
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


static inline void riscvv_ifft8x8_real(
	float block[restrict static 1],
	uint32_t column_count)
{
	float w[8 * column_count];

	real_to_complex_NxNc(block, w, column_count, 8);

	ifftN_to_ifft2N(block, 32, w, 8, column_count);

	ifft4xNc(w, block, column_count);

}

//--16x16--------------------------------------------------------------

static inline void riscvv_fft16x16_real(
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


static inline void riscvv_ifft16x16_real(float block[restrict static 256], size_t column_count){
	
	float w[16 * column_count]; //todo make in place

	real_to_complex_NxNc(block, w, column_count, 16);

	ifftN_to_ifft2N(block, 128, w, 16, column_count);

	riscvv_ifft8xNr(w, block, column_count);

}