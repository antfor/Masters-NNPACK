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
	//todo vectorize
	const uint32_t simd_width = nnp_hwinfo.simd_width;
	const uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	for (size_t i = 0; i < HALF_BLOCK_LENGTH/jump; i ++) {

		for(int j = 0; j < jump; j++){
			int ind = i*jump + j + (i*jump + j)/BLOCK_SIZE * BLOCK_SIZE;
			transform[j + 0] =    block[ind + 0];
			transform[j + jump] = block[ind + BLOCK_SIZE];
			
		}
		transform += transform_stride;
	}

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
