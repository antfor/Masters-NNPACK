#pragma once

#include <nnpack/fft-constants.h>
#include <sve/fft/soa.h>
#include <arm_sve.h>
#include <nnpack/hwinfo.h>


static inline void sve_fft8x8_dualreal_128(float block[restrict static 16]){
	float w[16] = {block[0], block[1],
				   block[2], block[3], 
				   block[8], block[9],
				   block[10], block[11],
				   block[4], block[5],
				   block[6], block[7],
				   block[12], block[13],
				   block[14], block[15]}; 

	block[0] = w[0];
	block[4] = w[4];
	
	block[1] = w[8];
	block[5] = w[12]; 
	
	block[2] = 0.5f * (w[1] + w[7]);
	block[6] = 0.5f * (w[9] - w[15]);
	
	block[3] = 0.5f * (w[9] + w[15]);
	block[7] = 0.5f * (w[7] - w[1]);
	
	block[8] = 0.5f * (w[2] + w[6]);
	block[12] = 0.5f * (w[10] - w[14]);
	
	block[9] = 0.5f * (w[10] + w[14]);
	block[13] = 0.5f * (w[6] - w[2]);
	
	block[10] = 0.5f * (w[3] + w[5]);
	block[14] = 0.5f * (w[11] - w[13]);
	
	block[11] = 0.5f * (w[11] + w[13]);
	block[15] = 0.5f * (w[5] - w[3]);

}

static inline void sve_fft8x8_dualreal(float block[restrict static 16]){
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

static inline void sve_ifft8x8_dualreal(const float transform[restrict static 1], size_t transform_stride, float block[restrict static 64]){
	
	// Load rows
	const uint32_t HALF_BLOCK_LENGTH = 32;
	const uint32_t simd_width = nnp_hwinfo.simd_width;

	svbool_t pg;
	const uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	const uint32_t jumps = (HALF_BLOCK_LENGTH + jump - 1)/jump; //round up
	const svbool_t vlen = svwhilelt_b32_s32(0, jump); 
	
	for(uint32_t i = 0; i < jumps; i++){
		pg = svwhilelt_b32_s32(i * jump, HALF_BLOCK_LENGTH);
		pg = svmov_z(pg, vlen);
		
		const svfloat32_t real = svld1(pg, transform); 
		const svfloat32_t imag = svld1(pg, transform + jump); 
		
		svst1(pg, block + i * jump + 0, real);
		svst1(pg, block + i * jump + HALF_BLOCK_LENGTH, imag);

		transform += transform_stride;
	}

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


//todo remove scalar

static inline void scalar_fft8_dualreal(
	const float seq[restrict static 16],
	float x0[restrict static 1],
	float y0[restrict static 1],
	float x1r[restrict static 1],
	float y1r[restrict static 1],
	float x2r[restrict static 1],
	float y2r[restrict static 1],
	float x3r[restrict static 1],
	float y3r[restrict static 1],
	float x4[restrict static 1],
	float y4[restrict static 1],
	float x1i[restrict static 1],
	float y1i[restrict static 1],
	float x2i[restrict static 1],
	float y2i[restrict static 1],
	float x3i[restrict static 1],
	float y3i[restrict static 1])
{
	float w0r, w1r, w2r, w3r, w4r, w5r, w6r, w7r;
	float w0i, w1i, w2i, w3i, w4i, w5i, w6i, w7i;
	scalar_fft8_soa(seq,
		&w0r, &w1r, &w2r, &w3r, &w4r, &w5r, &w6r, &w7r,
		&w0i, &w1i, &w2i, &w3i, &w4i, &w5i, &w6i, &w7i);

	*x0  = w0r;
	*y0  = w0i;
	*x1r = 0.5f * (w1r + w7r);
	*y1r = 0.5f * (w1i + w7i);
	*x2r = 0.5f * (w2r + w6r);
	*y2r = 0.5f * (w2i + w6i);
	*x3r = 0.5f * (w3r + w5r);
	*y3r = 0.5f * (w3i + w5i);

	*x4  = w4r;
	*y4  = w4i;
	*x1i = 0.5f * (w1i - w7i);
	*y1i = 0.5f * (w7r - w1r);
	*x2i = 0.5f * (w2i - w6i);
	*y2i = 0.5f * (w6r - w2r);
	*x3i = 0.5f * (w3i - w5i);
	*y3i = 0.5f * (w5r - w3r);
}

static inline void scalar_fft16_dualreal(
	const float seq[restrict static 32],
	float x0[restrict static 1],
	float y0[restrict static 1],
	float x1r[restrict static 1],
	float y1r[restrict static 1],
	float x2r[restrict static 1],
	float y2r[restrict static 1],
	float x3r[restrict static 1],
	float y3r[restrict static 1],
	float x4r[restrict static 1],
	float y4r[restrict static 1],
	float x5r[restrict static 1],
	float y5r[restrict static 1],
	float x6r[restrict static 1],
	float y6r[restrict static 1],
	float x7r[restrict static 1],
	float y7r[restrict static 1],
	float x8[restrict static 1],
	float y8[restrict static 1],
	float x1i[restrict static 1],
	float y1i[restrict static 1],
	float x2i[restrict static 1],
	float y2i[restrict static 1],
	float x3i[restrict static 1],
	float y3i[restrict static 1],
	float x4i[restrict static 1],
	float y4i[restrict static 1],
	float x5i[restrict static 1],
	float y5i[restrict static 1],
	float x6i[restrict static 1],
	float y6i[restrict static 1],
	float x7i[restrict static 1],
	float y7i[restrict static 1])
{
	float w0r, w1r, w2r, w3r, w4r, w5r, w6r, w7r, w8r, w9r, w10r, w11r, w12r, w13r, w14r, w15r;
	float w0i, w1i, w2i, w3i, w4i, w5i, w6i, w7i, w8i, w9i, w10i, w11i, w12i, w13i, w14i, w15i;
	scalar_fft16_soa(seq,
		&w0r, &w1r, &w2r, &w3r, &w4r, &w5r, &w6r, &w7r, &w8r, &w9r, &w10r, &w11r, &w12r, &w13r, &w14r, &w15r,
		&w0i, &w1i, &w2i, &w3i, &w4i, &w5i, &w6i, &w7i, &w8i, &w9i, &w10i, &w11i, &w12i, &w13i, &w14i, &w15i);

	*x0  = w0r;
	*y0  = w0i;
	*x1r = 0.5f * (w1r + w15r);
	*y1r = 0.5f * (w1i + w15i);
	*x2r = 0.5f * (w2r + w14r);
	*y2r = 0.5f * (w2i + w14i);
	*x3r = 0.5f * (w3r + w13r);
	*y3r = 0.5f * (w3i + w13i);
	*x4r = 0.5f * (w4r + w12r);
	*y4r = 0.5f * (w4i + w12i);
	*x5r = 0.5f * (w5r + w11r);
	*y5r = 0.5f * (w5i + w11i);
	*x6r = 0.5f * (w6r + w10r);
	*y6r = 0.5f * (w6i + w10i);
	*x7r = 0.5f * (w7r + w9r);
	*y7r = 0.5f * (w7i + w9i);

	*x8  = w8r;
	*y8  = w8i;
	*x1i = 0.5f * (w1i - w15i);
	*y1i = 0.5f * (w15r - w1r);
	*x2i = 0.5f * (w2i - w14i);
	*y2i = 0.5f * (w14r - w2r);
	*x3i = 0.5f * (w3i - w13i);
	*y3i = 0.5f * (w13r - w3r);
	*x4i = 0.5f * (w4i - w12i);
	*y4i = 0.5f * (w12r - w4r);
	*x5i = 0.5f * (w5i - w11i);
	*y5i = 0.5f * (w11r - w5r);
	*x6i = 0.5f * (w6i - w10i);
	*y6i = 0.5f * (w10r - w6r);
	*x7i = 0.5f * (w7i - w9i);
	*y7i = 0.5f * (w9r - w7r);
}

static inline void scalar_ifft8_dualreal(
	float x0, float y0, float x1r, float y1r, float x2r, float y2r, float x3r, float y3r,
	float x4, float y4, float x1i, float y1i, float x2i, float y2i, float x3i, float y3i,
	float seq[restrict static 16])
{
	float w0r = x0;
	float w0i = y0;
	float w1r = x1r - y1i;
	float w1i = x1i + y1r;
	float w2r = x2r - y2i;
	float w2i = x2i + y2r;
	float w3r = x3r - y3i;
	float w3i = x3i + y3r;

	float w4r = x4;
	float w4i = y4;
	float w5r = y3i + x3r;
	float w5i = y3r - x3i;
	float w6r = y2i + x2r;
	float w6i = y2r - x2i;
	float w7r = y1i + x1r;
	float w7i = y1r - x1i;

	scalar_ifft8_soa(
		w0r, w1r, w2r, w3r, w4r, w5r, w6r, w7r,
		w0i, w1i, w2i, w3i, w4i, w5i, w6i, w7i,
		seq);
}

static inline void scalar_ifft16_dualreal(
	float x0,  float y0,  float x1r, float y1r, float x2r, float y2r, float x3r, float y3r,
	float x4r, float y4r, float x5r, float y5r, float x6r, float y6r, float x7r, float y7r,
	float x8,  float y8,  float x1i, float y1i, float x2i, float y2i, float x3i, float y3i,
	float x4i, float y4i, float x5i, float y5i, float x6i, float y6i, float x7i, float y7i,
	float seq[restrict static 16])
{
	float w0r = x0;
	float w0i = y0;
	float w1r = x1r - y1i;
	float w1i = x1i + y1r;
	float w2r = x2r - y2i;
	float w2i = x2i + y2r;
	float w3r = x3r - y3i;
	float w3i = x3i + y3r;
	float w4r = x4r - y4i;
	float w4i = x4i + y4r;
	float w5r = x5r - y5i;
	float w5i = x5i + y5r;
	float w6r = x6r - y6i;
	float w6i = x6i + y6r;
	float w7r = x7r - y7i;
	float w7i = x7i + y7r;

	float w8r  = x8;
	float w8i  = y8;
	float w9r  = y7i + x7r;
	float w9i  = y7r - x7i;
	float w10r = y6i + x6r;
	float w10i = y6r - x6i;
	float w11r = y5i + x5r;
	float w11i = y5r - x5i;
	float w12r = y4i + x4r;
	float w12i = y4r - x4i;
	float w13r = y3i + x3r;
	float w13i = y3r - x3i;
	float w14r = y2i + x2r;
	float w14i = y2r - x2i;
	float w15r = y1i + x1r;
	float w15i = y1r - x1i;

	scalar_ifft16_soa(
		w0r, w1r, w2r, w3r, w4r, w5r, w6r, w7r, w8r, w9r, w10r, w11r, w12r, w13r, w14r, w15r,
		w0i, w1i, w2i, w3i, w4i, w5i, w6i, w7i, w8i, w9i, w10i, w11i, w12i, w13i, w14i, w15i,
		seq);
}