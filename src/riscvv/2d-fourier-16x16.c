#include <stdint.h>
#include <stddef.h>

#include <riscvv/fft/real.h>
#include <riscvv/fft/dualreal.h>

#include <nnpack/utils.h>
#include <nnpack/activations.h>


#define BLOCK_SIZE 16


void nnp_fft16x16_with_offset__scalar(
	const float data[restrict static 1],
	float transform[restrict static 1],
	size_t data_stride, size_t transform_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);
	
}

#if !NNP_INFERENCE_ONLY
void nnp_ifft16x16_with_offset__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count,
	uint32_t row_offset, uint32_t column_offset)
{
	transform_stride /= sizeof(float);


}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_ifft16x16_with_bias__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	
}

void nnp_ifft16x16_with_bias_with_relu__scalar(
	const float transform[restrict static 1],
	float data[restrict static 1],
	const float bias[restrict static 1],
	size_t transform_stride, size_t data_stride,
	uint32_t row_count, uint32_t column_count)
{
	transform_stride /= sizeof(float);

	
}
