#include <stdint.h>
#include <stddef.h>

#include <riscvv/fft/real.h>
#include <riscvv/fft/dualreal.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>

#include <riscvv/fft/rv-printf.h>

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

	riscvv_fft8xN_real(row0, row4, data_stride, row_offset, row_count, column_offset, column_count, block);

	riscvv_fft8x8_complex(block);

	riscvv_fft8x8_dualreal(block);

	//store
	const uint64_t simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	uint32_t jump = imin(HALF_BLOCK_LENGTH, simd_width);
	uint32_t gvl = __builtin_epi_vsetvl(jump, __epi_e32, __epi_m1);
	uint32_t jumps = idiv_ceil(HALF_BLOCK_LENGTH, jump);

	__epi_2xf32 real, imag;
	__epi_2xi32 ind_load = rvindex_adress(0,2,gvl); 

	for(uint32_t i = 0; i < jumps; i++){

		real = __builtin_epi_vload_indexed_2xf32(block + i * jump * 2 + 0, ind_load, gvl);
		imag = __builtin_epi_vload_indexed_2xf32(block + i * jump * 2 + 1, ind_load, gvl);

		__builtin_epi_vstore_2xf32(transform + 0, real, gvl);
		__builtin_epi_vstore_2xf32(transform + jump, imag, gvl);
		transform += transform_stride;
	}

}


void nnp_ifft8x8_with_bias__riscvv(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE*BLOCK_SIZE];

	riscvv_ifft8x8_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 64.0f;

	riscvv_ifft8x8_complex(block);

	riscvv_ifft8x8_real(block, column_count);

	//todo vectorize
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = block[row + column * BLOCK_SIZE];
		}
	}

}


void nnp_ifft8x8_with_bias_with_relu__riscvv(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	float block[BLOCK_SIZE*BLOCK_SIZE];

	riscvv_ifft8x8_dualreal(transform, transform_stride, block);

	block[0] += (*bias) * 64.0f;

	riscvv_ifft8x8_complex(block);

	riscvv_ifft8x8_real(block, column_count);

	//todo vectorize
	for (size_t row = 0; row < row_count; row++) {
		for (size_t column = 0; column < column_count; column++) {
			data[row * data_stride + column] = relu(block[row + column * BLOCK_SIZE],0);
		}
	}

}

//--scalar-----------------------------------------------------


void nnp_fft8x8_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	nnp_fft8x8_with_offset__riscvv(data, transform, data_stride, transform_stride, row_count, column_count, row_offset, column_offset);
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft8x8_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{

}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft8x8_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	
	nnp_ifft8x8_with_bias__riscvv(transform, data, bias, transform_stride, data_stride, row_count, column_count);
}

void nnp_ifft8x8_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	
	nnp_ifft8x8_with_bias_with_relu__riscvv(transform, data, bias, transform_stride, data_stride, row_count, column_count);
}

