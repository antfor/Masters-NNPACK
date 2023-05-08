#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>
#include <riscvv/fft/fft-util.h>

inline static void aos4_offset(const bool *n, size_t stride, svuint32_t *t_n_offset)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

	__uint32_t index_input[4];
	index_input[0] = 0;
	index_input[1] = cSum_n[0] * stride * 4;
	index_input[2] = cSum_n[1] * stride * 4;
	index_input[3] = cSum_n[2] * stride * 4;
	// Index 4
    *t_n_offset = indexA(index_input, 4, 1 * 4);
}

inline static void aos4_pred_and_offset(uint32_t row_start, uint32_t row_count,  size_t stride, __epi_2xi1 *mask_a, __epi_2xi1 *mask_b, __epi_2xi32 *t_lo_offset, __epi_2xi32 *t_hi_offset, long gvl)
{
    const uint32_t row_end = row_start + row_count;

    const bool a[4] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3};
    const bool b[4] = {row_start <= 4 && row_end > 4, row_start <= 5 && row_end > 5, row_start <= 6 && row_end > 6, row_start <= 7 && row_end > 7};

    bool no_jump[8];
    no_jump[0] = 1;
    no_jump[1] = no_jump[0] && !(a[0] && --row_count == 0);
    no_jump[2] = no_jump[1] && !(b[0] && --row_count == 0);
    no_jump[3] = no_jump[2] && !(a[1] && --row_count == 0);
    no_jump[4] = no_jump[3] && !(b[1] && --row_count == 0);
    no_jump[5] = no_jump[4] && !(a[2] && --row_count == 0);
    no_jump[6] = no_jump[5] && !(b[2] && --row_count == 0);
    no_jump[7] = no_jump[6] && !(a[3] && --row_count == 0);

	__uint32_t mask_a_quad[4];
	mask_a_quad[0] = a[0] && no_jump[0];
	mask_a_quad[1] = a[1] && no_jump[2];
	mask_a_quad[2] = a[2] && no_jump[4];
	mask_a_quad[3] = a[3] && no_jump[6];

	__uint32_t mask_b_quad[4];
	mask_b_quad[0] = b[0] && no_jump[1];
	mask_b_quad[1] = b[1] && no_jump[3];
	mask_b_quad[2] = b[2] && no_jump[5];
	mask_b_quad[3] = b[3] && no_jump[7];

    *mask_a = __builtin_epi_cast_2xi1_2xi32(dupq(mask_a_quad, gvl));
    *mask_b = __builtin_epi_cast_2xi1_2xi32(dupq(mask_b_quad, gvl));

    aos4_offset(a, stride, t_lo_offset);
    aos4_offset(b, stride, t_hi_offset);
}