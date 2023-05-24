#pragma once

#include <stdio.h>

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
