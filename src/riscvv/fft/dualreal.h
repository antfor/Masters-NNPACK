#pragma once

#include <stdio.h>
#include <riscvv/fft/rv-printf.h>

//--load--------------------------------------------------------------

static inline void riscvv_load(const uint32_t HALF_BLOCK_LENGTH, const float transform[restrict static 1], size_t transform_stride, float block[restrict static 1]){

	// Load rows
	const uint32_t simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); //nnp_hwinfo.simd_width;

	const uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	long gvl = __builtin_epi_vsetvl(jump, __epi_e32, __epi_m1);

	const uint32_t jumps = (HALF_BLOCK_LENGTH + jump - 1)/jump; //round up
	
	__epi_2xf32 real, imag;

	for(uint32_t i = 0; i < jumps; i++){
		
		real = __builtin_epi_vload_2xf32(transform, gvl); 
		imag = __builtin_epi_vload_2xf32(transform + jump, gvl); 
		
		__builtin_epi_vstore_2xf32(block + i * jump + 0, real, gvl);
		__builtin_epi_vstore_2xf32(block + i * jump + HALF_BLOCK_LENGTH, imag, gvl);

		transform += transform_stride;
	}

}

//--fft8x8--------------------------------------------------------------

static inline void riscvv_fft8x8_dualreal(float block[restrict static 16]){
	float w[16] = {block[0], block[2],
				   block[4], block[6], 
				   block[8], block[10],
				   block[12], block[14],
				   block[1], block[3],
				   block[5], block[7],
				   block[9], block[11],
				   block[13], block[15]}; 

	block[0] = w[0];
	block[1] = w[4];
	
	block[2] = w[8];
	block[3] = w[12]; 
	
	block[4] = 0.5f * (w[1] + w[7]);
	block[5] = 0.5f * (w[9] - w[15]);
	
	block[6] = 0.5f * (w[9] + w[15]);
	block[7] = 0.5f * (w[7] - w[1]);
	
	block[8] = 0.5f * (w[2] + w[6]);
	block[9] = 0.5f * (w[10] - w[14]);
	
	block[10] = 0.5f * (w[10] + w[14]);
	block[11] = 0.5f * (w[6] - w[2]);
	
	block[12] = 0.5f * (w[3] + w[5]);
	block[13] = 0.5f * (w[11] - w[13]);
	
	block[14] = 0.5f * (w[11] + w[13]);
	block[15] = 0.5f * (w[5] - w[3]);
}


static inline void riscvv_ifft8x8_dualreal(const float transform[restrict static 1], size_t transform_stride, float block[restrict static 64]){
	
	// Load rows
	const uint32_t HALF_BLOCK_LENGTH = 32;
	riscvv_load(HALF_BLOCK_LENGTH, transform, transform_stride, block);
	
	// stuff
	float x0 = block[0 + 0];
	float x4 = block[0 + HALF_BLOCK_LENGTH]; 
	float y0 = block[1 + 0];
	float y4 = block[1 + HALF_BLOCK_LENGTH];
	float x1r = block[2 + 0];
	float x1i = block[2 + HALF_BLOCK_LENGTH];
	float y1r = block[3 + 0];
	float y1i = block[3 + HALF_BLOCK_LENGTH];

	float x2r = block[4 + 0];
	float x2i = block[4 + HALF_BLOCK_LENGTH];
	float y2r = block[5 + 0];
	float y2i = block[5 + HALF_BLOCK_LENGTH];
	float x3r = block[6 + 0];
	float x3i = block[6 + HALF_BLOCK_LENGTH];
	float y3r = block[7 + 0];
	float y3i = block[7 + HALF_BLOCK_LENGTH];
	

	block[0 + 0] = x0;
	block[0 + HALF_BLOCK_LENGTH] = y0;
	block[1 + 0] = x1r - y1i;
	block[1 + HALF_BLOCK_LENGTH] = x1i + y1r;
	block[2 + 0] = x2r - y2i;
	block[2 + HALF_BLOCK_LENGTH] = x2i + y2r;
	block[3 + 0] = x3r - y3i;
	block[3 + HALF_BLOCK_LENGTH] = x3i + y3r;
	block[4 + 0] = x4;
	block[4 + HALF_BLOCK_LENGTH] = y4;
	block[5 + 0] = y3i + x3r;
	block[5 + HALF_BLOCK_LENGTH] = y3r - x3i;
	block[6 + 0] = y2i + x2r;
	block[6 + HALF_BLOCK_LENGTH] = y2r - x2i;
	block[7 + 0] = y1i + x1r;
	block[7 + HALF_BLOCK_LENGTH] = y1r - x1i;
	

}
//--fft16x16--------------------------------------------------------------

static inline void dualreal_Nbyte(float tf[restrict static 32], int N, int offset, int channels, int BLOCK_LENGTH){

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
    const int numVals = (simd_width * 2) / N;

    int pg = imin(simd_width, N * channels / 2);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 xr_r, xNr_r, Fr_r, Gr_r;
	__epi_2xf32 xr_i, xNr_i, Fr_i, Gr_i;
	__epi_2xf32 half =  __builtin_epi_vfmv_v_f_2xf32(0.5f, gvl);
	__epi_2xf32 half_conjugate = __builtin_epi_vfmv_v_f_2xf32(-0.5f, gvl);
	__epi_2xf32 minus_one =  __builtin_epi_vfmv_v_f_2xf32(-1.0f, gvl);

	__epi_2xi32 indr = indexN_address(gvl, 0, 1, BLOCK_LENGTH, N/2);
	__epi_2xi32 indN_r = indexN_address(gvl, N, -1, BLOCK_LENGTH, N/2);

	__epi_2xi32 ind_store = indexN_address(gvl, 0, 2, BLOCK_LENGTH, N/2);

	for(int channel = 0; channel < channels; channel+=numVals){

		//load
		xr_r = __builtin_epi_vload_indexed_2xf32(tf + channel * BLOCK_LENGTH + 0, indr, gvl);
		xr_i = __builtin_epi_vload_indexed_2xf32(tf + channel * BLOCK_LENGTH + N, indr, gvl);
		xNr_r = __builtin_epi_vload_indexed_2xf32(tf + channel * BLOCK_LENGTH + 0, indN_r, gvl);
		xNr_i = __builtin_epi_vload_indexed_2xf32(tf + channel * BLOCK_LENGTH + N, indN_r, gvl);

		xr_r = __builtin_epi_vfmul_2xf32(xr_r, half, gvl);
		xr_i = __builtin_epi_vfmul_2xf32(xr_i, half, gvl);

		xNr_r = __builtin_epi_vfmul_2xf32(xNr_r, half, gvl);
		xNr_i = __builtin_epi_vfmul_2xf32(xNr_i, half_conjugate, gvl);

		Fr_r = __builtin_epi_vfadd_2xf32(xNr_r, xr_r, gvl);
		Fr_i = __builtin_epi_vfadd_2xf32(xNr_i, xr_i, gvl);

		Gr_r = __builtin_epi_vfsub_2xf32(xNr_i, xr_i, gvl);
		Gr_r = __builtin_epi_vfmul_2xf32(Gr_r, minus_one, gvl);
		Gr_i = __builtin_epi_vfsub_2xf32(xNr_r, xr_r, gvl);

		//store
		__builtin_epi_vstore_indexed_2xf32(tf + channel * BLOCK_LENGTH + 0,Fr_r, ind_store ,gvl);
		__builtin_epi_vstore_indexed_2xf32(tf + channel * BLOCK_LENGTH + 1,Gr_r, ind_store ,gvl);
		__builtin_epi_vstore_indexed_2xf32(tf + channel * BLOCK_LENGTH + N + 0,Fr_i, ind_store ,gvl);
		__builtin_epi_vstore_indexed_2xf32(tf + channel * BLOCK_LENGTH + N + 1,Gr_i, ind_store ,gvl);

	}

}


static inline void riscvv_fft16x16_dualreal(float tf[restrict static 32]){

	float x0 = tf[0 + 0];
	float y0 = tf[0 + 16];
	float x8 = tf[8 + 0];
	float y8 = tf[8 + 16];

	dualreal_Nbyte(tf, 16, 2, 1, 256);

	tf[0 + 0] = x0;
	tf[0 + 1] = y0;
	tf[0 + 16] = x8;
	tf[1 + 16] = y8;

}


static inline void idualreal_Nbyte(float tf[restrict static 32], int N, int offset){

	const uint32_t HALF_BLOCK_LENGTH = N * N /2;

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2);

	int pg = imin(simd_width, (N - offset)/2);
	long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 xr_r, xNr_r, x_r, y_r;
	__epi_2xf32 xr_i, xNr_i, x_i, y_i;

	__epi_2xi32 ind_load = rvindex_adress(0 + offset, 2, gvl);
	__epi_2xi32 ind_store_y = rvindex_adress(N-1, -1, gvl);


	for(int row = 0; row < N; row+=numVals){

		//load
		xr_r = __builtin_epi_vload_indexed_2xf32(tf + row + 0, ind_load, gvl);
		xr_i = __builtin_epi_vload_indexed_2xf32(tf + row + 1, ind_load, gvl);

		xNr_r = __builtin_epi_vload_indexed_2xf32(tf + row + HALF_BLOCK_LENGTH + 0, ind_load, gvl);
		xNr_i = __builtin_epi_vload_indexed_2xf32(tf + row + HALF_BLOCK_LENGTH + 1, ind_load, gvl);

		x_r =__builtin_epi_vfsub_2xf32(xr_r, xNr_i, gvl); 
		x_i =__builtin_epi_vfadd_2xf32(xr_i, xNr_r, gvl);
		y_r =__builtin_epi_vfadd_2xf32(xr_r, xNr_i, gvl);
		y_i =__builtin_epi_vfsub_2xf32(xr_i, xNr_r, gvl);

		//store
		__builtin_epi_vstore_2xf32(tf + row + offset/2 + 0, x_r, gvl);
		__builtin_epi_vstore_2xf32(tf + row + offset/2 + HALF_BLOCK_LENGTH, x_i, gvl);
		__builtin_epi_vstore_indexed_2xf32(tf + row + 0, y_r, ind_store_y,gvl);
		__builtin_epi_vstore_indexed_2xf32(tf + row + HALF_BLOCK_LENGTH, y_i, ind_store_y,gvl);

	}

}

static inline void riscvv_ifft16x16_dualreal(const float transform[restrict static 1], size_t transform_stride, float f[restrict static 256]){

	// Load rows
	const uint32_t HALF_BLOCK_LENGTH = 128;

	riscvv_load(HALF_BLOCK_LENGTH, transform, transform_stride, f);

	float y0 = f[1];
	float x8 = f[0 + HALF_BLOCK_LENGTH];
	float y8 = f[1 + HALF_BLOCK_LENGTH];

	idualreal_Nbyte(f, 16, 2);

	f[0 + HALF_BLOCK_LENGTH] = y0;
	f[8] = x8;
	f[8 + HALF_BLOCK_LENGTH] = y8;
}