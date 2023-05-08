#include <stdint.h>
#include <stddef.h>

#include <riscvv/fft/real.h>
#include <riscvv/fft/soa.h>
#include <riscvv/fft/dualreal.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>

#include <nnpack/hwinfo.h>

#define BLOCK_SIZE 8
#define BLOCK_LENGTH 64
#define HALF_BLOCK_LENGTH 32


void nnp_fft8x8_with_offset__riscvv(
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

	riscvv_fft8xN_real(row0, row4, data_stride, row_offset, row_count, &block[column_offset], BLOCK_SIZE, column_count);

	complex_256(block, transform, transform_stride);
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft8x8_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];
	{
		const float x0 = transform[0];
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row_offset + row][column_offset + column];
		}
	}
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft8x8_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];

	const float bias_value = *bias;
	{
		const float x0 = transform[0] + bias_value * 64.0f;
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row][column];
		}
	}
}

void nnp_ifft8x8_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE][BLOCK_SIZE];

	const float bias_value = *bias;
	{
		const float x0 = transform[0] + bias_value * 64.0f;
		const float x4 = transform[1];
		transform += transform_stride;
		const float y0 = transform[0];
		const float y4 = transform[1];
		transform += transform_stride;
		const float x1r = transform[0];
		const float x1i = transform[1];
		transform += transform_stride;
		const float y1r = transform[0];
		const float y1i = transform[1];
		transform += transform_stride;
		const float x2r = transform[0];
		const float x2i = transform[1];
		transform += transform_stride;
		const float y2r = transform[0];
		const float y2i = transform[1];
		transform += transform_stride;
		const float x3r = transform[0];
		const float x3i = transform[1];
		transform += transform_stride;
		const float y3r = transform[0];
		const float y3i = transform[1];
		transform += transform_stride;
		scalar_ifft8_dualreal(
			x0, y0, x1r, y1r, x2r, y2r, x3r, y3r,
			x4, y4, x1i, y1i, x2i, y2i, x3i, y3i,
			&block[0][0]);
	}
	for (uint32_t row = 2; row < BLOCK_SIZE; row += 2) {
		const float f0r = transform[0];
		const float f0i = transform[1];
		transform += transform_stride;
		const float f1r = transform[0];
		const float f1i = transform[1];
		transform += transform_stride;
		const float f2r = transform[0];
		const float f2i = transform[1];
		transform += transform_stride;
		const float f3r = transform[0];
		const float f3i = transform[1];
		transform += transform_stride;
		const float f4r = transform[0];
		const float f4i = transform[1];
		transform += transform_stride;
		const float f5r = transform[0];
		const float f5i = transform[1];
		transform += transform_stride;
		const float f6r = transform[0];
		const float f6i = transform[1];
		transform += transform_stride;
		const float f7r = transform[0];
		const float f7i = transform[1];
		transform += transform_stride;
		scalar_ifft8_soa(
			f0r, f1r, f2r, f3r, f4r, f5r, f6r, f7r,
			f0i, f1i, f2i, f3i, f4i, f5i, f6i, f7i,
			&block[row][0]);
	}

	for (uint32_t column = 0; column < BLOCK_SIZE; column++) {
		const float f0  = block[0][column];
		const float f4  = block[1][column];
		const float f1r = block[2][column];
		const float f1i = block[3][column];
		const float f2r = block[4][column];
		const float f2i = block[5][column];
		const float f3r = block[6][column];
		const float f3i = block[7][column];
		scalar_ifft8_real(
			f0, f4, f1r, f1i, f2r, f2i, f3r, f3i,
			&block[0][column], &block[BLOCK_SIZE / 2][column],
			BLOCK_SIZE);
	}

	for (uint32_t row = 0; row < row_count; row++) {
		for (uint32_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[row][column], 0.0f);
		}
	}
}
