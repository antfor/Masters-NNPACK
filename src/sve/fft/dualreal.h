#pragma once

#include <nnpack/fft-constants.h>
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

static inline void sve_load(const uint32_t HALF_BLOCK_LENGTH, const float transform[restrict static 1], size_t transform_stride, float block[restrict static 1]){

	// Load rows
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

}

static inline void sve_ifft8x8_dualreal(const float transform[restrict static 1], size_t transform_stride, float block[restrict static 64]){
	
	// Load rows
	const uint32_t HALF_BLOCK_LENGTH = 32;
	sve_load(HALF_BLOCK_LENGTH, transform, transform_stride, block);

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

//-16------------------------------------------------------------------------

static inline void dualreal_Nbyte(float tf[restrict static 32], int N, int offset){


	svbool_t pg;
	const svbool_t all = svptrue_b32();

	const svfloat32_t half_conjugate = svdupq_f32(0.5f,-0.5f, 0.5f,-0.5f);
	const svfloat32_t i = svdupq_f32(0.0f, 1.0f, 0.0f, 1.0f);
	svfloat32_t xr, xN_r, Fr, Gr, dif;
	size_t numVals = svcntw()/N;

	svuint32_t indr =   index2(0 + offset/2, N + offset/2, 1);
	svuint32_t indN_r = index2(N -1, N -1 + N, -1);

	for(int row = 0; row < 1; row+=numVals){
		
		pg = svwhilelt_b32_s32(row + offset, N);
		//load
		xr = svld1_gather_index(pg, tf + row, indr);
		xr = svmul_m(pg, xr, 0.5f);

		xN_r = svld1_gather_index(pg, tf + row, indN_r);
		xN_r = svmul_m(pg, xN_r, half_conjugate);

		Fr = svadd_m(pg, xN_r, xr);

		svfloat32_t dif  = svsub_m(pg, xN_r, xr);
		cmul_twiddle(&pg, &dif, &i, &Gr);

		//store
		svst1(pg, tf + row + 0  + offset, svtrn1(Fr, Gr));
		svst1(pg, tf + row + N + offset, svtrn2(Fr, Gr));
	}
}

static inline void sve_fft16x16_dualreal(float tf[restrict static 32]){

	float y0 = tf[0 + 16];
	float x8 = tf[8 + 0];
	float y8 = tf[8 + 16];

	dualreal_Nbyte(tf, 16, 2);

	tf[0 + 1] = y0;
	tf[0 + 16] = x8;
	tf[1 + 16] = y8;

}

static inline void idualreal_Nbyte(float tf[restrict static 32], int N, int offset){


	svbool_t pg;
	const uint32_t HALF_BLOCK_LENGTH = N * N /2;

	svfloat32_t xr, xN_r, x, y;
	size_t numVals = svcntw();

	svuint32_t indr =  svindex_u32(0 + offset, 1);
	const svbool_t all = svptrue_b32();
	svuint32_t indN_r = svadd_m(all, indr, HALF_BLOCK_LENGTH);

	svuint32_t ind_store_x = index2(0, HALF_BLOCK_LENGTH, 1);
	svuint32_t ind_store_y = index2(N-1, N-1 + HALF_BLOCK_LENGTH, -1);


	for(int row = 0; row < N; row+=numVals){
		
		pg = svwhilelt_b32_s32(row + offset, N);
		
		//load
		xr = svld1_gather_index(pg, tf + row, indr);
		xN_r = svld1_gather_index(pg, tf + row, indN_r);

		x = svcadd_m(pg, xr, xN_r, 90);
		y = svcadd_m(pg, xr, xN_r, 270);

		//store 
		//todo zip x,y for better locality?
		svst1_scatter_index(pg, tf + row + offset/2, ind_store_x, x);
		svst1_scatter_index(pg, tf + row + 0, ind_store_y, y);
	}
}




static inline void sve_ifft16x16_dualreal(const float transform[restrict static 1], size_t transform_stride, float f[restrict static 256]){

	// Load rows
	const uint32_t HALF_BLOCK_LENGTH = 128;

	sve_load(HALF_BLOCK_LENGTH, transform, transform_stride, f);

	float y0 = f[1];
	float x8 = f[0 + HALF_BLOCK_LENGTH];
	float y8 = f[1 + HALF_BLOCK_LENGTH];

	idualreal_Nbyte(f, 16, 2);

	f[0 + HALF_BLOCK_LENGTH] = y0;

	f[8] = x8;
	f[8 + HALF_BLOCK_LENGTH] = y8;
}