#include <stdint.h>
#include <stddef.h>
#include <arm_sve.h>
#include <stdio.h>

#include <sve/fft/real.h>
#include <sve/fft/complex-soa.h>
#include <sve/fft/complex-channel-soa.h>
#include <sve/fft/dualreal.h>

#include <sve/fft/sve-print.h>
#include <nnpack/utils.h>
#include <nnpack/activations.h>


#define BLOCK_SIZE 16
#define BLOCK_LENGTH 256
#define HALF_BLOCK_LENGTH 128


void nnp_fft16x16_with_offset__sve(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);
	float block[BLOCK_LENGTH] = {0.0f};

	const float *restrict row0 = data;
	const float *restrict row8 = data + doz(BLOCK_SIZE / 2, row_offset) * data_stride;

	sve_fft16x16_real(row0, row8, data_stride,row_offset, row_count, column_offset, column_count, block);

	sve_fft16x16_complex(block);

	sve_fft16x16_dualreal(block);

	//store 
	const uint32_t simd_width = nnp_hwinfo.simd_width;
	const uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	const svbool_t all = svptrue_b32();//svwhilelt_b32_s32(0, jump);
	svfloat32_t real, imag;
	const svuint32_t ind_load = indexN(all, 0, 1, BLOCK_SIZE * 2, BLOCK_SIZE);

	for (size_t i = 0; i < HALF_BLOCK_LENGTH/jump; i ++) {

		real = svld1_gather_index(all, block + i * jump * 2 + 0, ind_load);
		imag = svld1_gather_index(all, block + i * jump * 2 + BLOCK_SIZE, ind_load);	

		svst1(all, transform + 0, real);
		svst1(all, transform + jump, imag);
		transform += transform_stride;
	}

}

void nnp_fft16x16_kernel__sve(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t channels, uint32_t transform_jump)
{
	uint32_t row_offset = 0;
	uint32_t column_offset = 0;

	transform_stride /= sizeof(float);
	transform_jump /= sizeof(float);

	float *block = (float*)calloc(BLOCK_LENGTH * channels, sizeof(float));

    float *row0 = (float *) data;
	float *row8 = (float *) data + doz(BLOCK_SIZE / 2, row_offset) * data_stride;


	sve_fft16x16_real_kernel(row0, row8, data_stride, row_count, column_count, block, channels);

	sve_fft16x16_complex_kernel(block, channels);

	sve_fft16x16_dualreal_kernel(block, channels);

	//store 
	const uint32_t simd_width = nnp_hwinfo.simd_width;
	const uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	const svbool_t all = svptrue_b32();//svwhilelt_b32_s32(0, jump);
	svfloat32_t real, imag;
	const svuint32_t ind_load = indexN(all, 0, 1, BLOCK_SIZE * 2, BLOCK_SIZE);

	float *transform_start = transform;
	float *block_start = block;

	for(int channel = 0; channel < channels; channel++){

		float * transform = transform_start;
		
		for (size_t i = 0; i < HALF_BLOCK_LENGTH/jump; i ++) {

			real = svld1_gather_index(all, block_start + i * jump * 2 + 0, ind_load);
			imag = svld1_gather_index(all, block_start + i * jump * 2 + BLOCK_SIZE, ind_load);	

			svst1(all, transform + 0, real);
			svst1(all, transform + jump, imag);
			transform += transform_stride;
		}
		transform_start += transform_jump;
		block_start += BLOCK_LENGTH;
	}
	

	free(block);
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft16x16_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{

}

#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft16x16_with_bias__sve(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	//printf("row_count: %d, column_count: %d\n", row_count, column_count);

	transform_stride /= sizeof(float);
	const int simd_width = nnp_hwinfo.simd_width;

	float block[BLOCK_SIZE*BLOCK_SIZE];

	sve_ifft16x16_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 256.0f;

	sve_ifft16x16_complex(block);

	sve_ifft16x16_real(block, column_count);

	//todo vectorize
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row + column * BLOCK_SIZE];
		}
	}

}

void nnp_ifft16x16_with_bias_with_relu__sve(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{

	transform_stride /= sizeof(float);
	const int simd_width = nnp_hwinfo.simd_width;

	float block[BLOCK_SIZE*BLOCK_SIZE];

	sve_ifft16x16_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 256.0f;

	sve_ifft16x16_complex(block);

	sve_ifft16x16_real(block, column_count);
	
	//todo vectorize
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[row + column * BLOCK_SIZE],0);
		}
	}
}


void nnp_fft16x16_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{

	nnp_fft16x16_with_offset__sve(data, transform, data_stride, transform_stride, row_count, column_count, row_offset, column_offset);

}


void nnp_ifft16x16_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	nnp_ifft16x16_with_bias__sve(transform, data, bias, transform_stride, data_stride, row_count, column_count);	
}



void nnp_ifft16x16_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	nnp_ifft16x16_with_bias_with_relu__sve(transform, data, bias, transform_stride, data_stride, row_count, column_count);
}
