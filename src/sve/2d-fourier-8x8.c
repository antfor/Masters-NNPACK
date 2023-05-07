#include <stdint.h>
#include <stddef.h>
#include <arm_sve.h>
#include <stdio.h>

#include <sve/fft/real.h>
#include <sve/fft/complex.h> 
#include <sve/fft/dualreal.h>
#include <sve/fft/sve-print.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>



#define BLOCK_SIZE 8
#define BLOCK_LENGTH 64
#define HALF_BLOCK_LENGTH 32

void complex_128(float block[BLOCK_LENGTH] ,float transform[restrict static 1], size_t transform_stride){
	
	sve_fft8x8_complex_128(block);
	sve_fft8x8_dualreal_128(block);

	//store
	const uint32_t simd_width = nnp_hwinfo.simd_width;
	float to_bytes = sizeof(float);
	const svuint32_t ind_store = indexN(svptrue_b32(),0, 1, 8, 4); 
	svbool_t pg;
	uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	uint32_t jumps = (HALF_BLOCK_LENGTH + jump - 1)/jump; //round up
	svbool_t vlen = svwhilelt_b32_s32(0, jump);
	
	for(uint32_t i = 0; i < jumps; i++){
		pg = svwhilelt_b32_s32(i * jump, HALF_BLOCK_LENGTH);
		pg = svmov_z(pg, vlen); 
		
		const svfloat32_t real = svld1_gather_index(pg, block + i * jump * 2 + 0 , ind_store); 
		const svfloat32_t imag = svld1_gather_index(pg, block + i * jump * 2 + BLOCK_SIZE/2 , ind_store); 

		svst1(pg, transform + 0, real);
		svst1(pg, transform + jump, imag);
		transform += transform_stride;
	}
}

void complex_256(float block[BLOCK_LENGTH] ,float transform[restrict static 1], size_t transform_stride){

	sve_fft8x8_complex(block);
	sve_fft8x8_dualreal(block);

	//store
	const uint32_t simd_width = nnp_hwinfo.simd_width;
	float to_bytes = sizeof(float);
	const svuint32_t ind_store = svindex_u32(0,to_bytes * 2); 
	svbool_t pg;
	uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	uint32_t jumps = (HALF_BLOCK_LENGTH + jump - 1)/jump; //round up
	svbool_t vlen = svwhilelt_b32_s32(0, jump);
	
	for(uint32_t i = 0; i < jumps; i++){
		pg = svwhilelt_b32_s32(i * jump, HALF_BLOCK_LENGTH);
		pg = svmov_z(pg, vlen); 
		
		const svfloat32_t real = svld1_gather_offset(pg, block + i * jump * 2 + 0 , ind_store); 
		const svfloat32_t imag = svld1_gather_offset(pg, block + i * jump * 2 + 1 , ind_store); 

		svst1(pg, transform + 0, real);
		svst1(pg, transform + jump, imag);
		transform += transform_stride;
	}
}


void nnp_fft8x8_with_offset__sve(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{

	transform_stride /= sizeof(float);

	float block[BLOCK_LENGTH] = {0.0f};

	const float *restrict row0 = data;
	const float *restrict row4 = data + doz(BLOCK_SIZE / 2, row_offset) * data_stride;

    sve_fft8xN_real(row0, row4, data_stride,row_offset, row_count, &block[column_offset], BLOCK_SIZE, column_count);

	complex_256(block, transform, transform_stride);
	/*
	if(nnp_hwinfo.simd_width % 2 == 0 && 0){
		complex_256(block, transform, transform_stride);

	}else{
		complex_128(block, transform, transform_stride);
	}
	*/
}


#if !NNP_INFERENCE_ONLY

void nnp_ifft8x8_with_offset__sve(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	
}

#endif /* !NNP_INFERENCE_ONLY */


void nnp_ifft8x8_with_bias__sve(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);
	const int simd_width = nnp_hwinfo.simd_width;

	float block[BLOCK_SIZE*BLOCK_SIZE];

	sve_ifft8x8_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 64.0f;

	sve_ifft8x8_complex(block);
	/*
	if(nnp_hwinfo.simd_width % 2 == 0 && 0){
		sve_ifft8x8_complex(block);
	}else{
		sve_ifft8x8_complex_128(block);
	}
	*/

	sve_ifft8x8_real(block, column_count);
	
	//todo vectorize
	const uint32_t rows[8] = {0,0,6,6,4,4,2,2};
	//real numbers
	for (size_t row = 0; row < row_count; row+=2) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[rows[row] * BLOCK_SIZE + 2 * column + 0];
		}
	}

	//imag numbers
	for (size_t row = 1; row < row_count; row+=2) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[rows[row] * BLOCK_SIZE + 2 * column + 1];
		}
	}

}

void nnp_ifft8x8_with_bias_with_relu__sve(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{

	transform_stride /= sizeof(float);
	const int simd_width = nnp_hwinfo.simd_width;

	float block[BLOCK_SIZE*BLOCK_SIZE];

	sve_ifft8x8_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 64.0f;

	sve_ifft8x8_complex(block);
	/*
	if(nnp_hwinfo.simd_width % 2 == 0 && 0){
		sve_ifft8x8_complex(block);
	}else{
		sve_ifft8x8_complex_128(block);
	}
	*/

	sve_ifft8x8_real(block, column_count);
	
	//todo vectorize
	const uint32_t rows[8] = {0,0,6,6,4,4,2,2};
	//real numbers
	for (size_t row = 0; row < row_count; row+=2) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[rows[row] * BLOCK_SIZE + 2 * column + 0], 0);
		}
	}

	//imag numbers
	for (size_t row = 1; row < row_count; row+=2) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[rows[row] * BLOCK_SIZE + 2 * column + 1], 0);
		}
	}

}



//--------------------------------------------------------------

void nnp_fft8x8_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	nnp_fft8x8_with_offset__sve(data, transform, data_stride, transform_stride, row_count, column_count, row_offset, column_offset);
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft8x8_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	nnp_ifft8x8_with_offset__sve(transform, data, transform_stride, data_stride, row_count, column_count, row_offset, column_offset);
}
#endif


void nnp_ifft8x8_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	nnp_ifft8x8_with_bias__sve(transform, data, bias, transform_stride, data_stride, row_count, column_count);	
}

void nnp_ifft8x8_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	nnp_ifft8x8_with_bias_with_relu__sve(transform, data, bias, transform_stride, data_stride, row_count, column_count);
}

