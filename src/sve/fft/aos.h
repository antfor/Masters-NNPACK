#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>
#include <arm_sve.h>
#include <stdbool.h>
#include <sve/fft/fft-util.h>


inline static void aos4_offset(const bool *n, size_t stride, svuint32_t *t_n_offset)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

    *t_n_offset = index4(0, cSum_n[0] * stride * 4, cSum_n[1] * stride * 4, cSum_n[2] * stride * 4, 1 * 4);
}

inline static void aos4_pred_and_offset(uint32_t row_start, uint32_t row_count, svbool_t *pg_a, svbool_t *pg_b, size_t stride, svuint32_t *t_lo_offset, svuint32_t *t_hi_offset)
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

    *pg_a = svdupq_b32(a[0] && no_jump[0], a[1] && no_jump[2], a[2] && no_jump[4], a[3] && no_jump[6]);
    *pg_b = svdupq_b32(b[0] && no_jump[1], b[1] && no_jump[3], b[2] && no_jump[5], b[3] && no_jump[7]);

    aos4_offset(a, stride, t_lo_offset);
    aos4_offset(b, stride, t_hi_offset);
}

inline static void aos8_offset(const bool *n, size_t stride, svuint32_t *t_n_offset){

	int cSum_n[7];
	cSum_n[0] = n[0];
	cSum_n[1] = cSum_n[0] + n[1];
	cSum_n[2] = cSum_n[1] + n[2];
	cSum_n[3] = cSum_n[2] + n[3];
	cSum_n[4] = cSum_n[3] + n[4];
	cSum_n[5] = cSum_n[4] + n[5];
	cSum_n[6] = cSum_n[5] + n[6];

	//*t_n_offset = index8(0, cSum_n[0] * stride * 4, cSum_n[1] * stride * 4, cSum_n[2] * stride * 4, cSum_n[3] * stride * 4, cSum_n[4] * stride * 4, cSum_n[5] * stride * 4, cSum_n[6] * stride * 4, 1 * 4);
    //fft8xNr_channel depend on jump now 4 it is 4
    *t_n_offset = indexA(svptrue_b32(), (uint32_t []){0,cSum_n[0] * stride * 4, cSum_n[1] * stride * 4, cSum_n[2] * stride * 4, cSum_n[3] * stride * 4, cSum_n[4] * stride * 4, cSum_n[5] * stride * 4, cSum_n[6] * stride * 4}, 8 ,1 * 4);
}


inline static void aos8_pred_and_offset(uint32_t row_start, uint32_t row_count, svbool_t *pg_a, svbool_t *pg_b, size_t stride, svuint32_t *t_lo_offset, svuint32_t *t_hi_offset)
{
	const uint32_t row_end = row_start + row_count;

    const bool a[8] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3, row_start <= 4, row_start <= 5, row_start <= 6, row_start <= 7};
    const bool b[8] = {row_start <= 8 && row_end > 8, row_start <= 9 && row_end > 9, row_start <= 10 && row_end > 10, row_start <= 11 && row_end > 11, row_start <= 12 && row_end > 12, row_start <= 13 && row_end > 13, row_start <= 14 && row_end > 14, row_start <= 15 && row_end > 15};

    bool no_jump[16];
    no_jump[0] = 1;
    no_jump[1] = no_jump[0] && !(a[0] && --row_count == 0);
    no_jump[2] = no_jump[1] && !(b[0] && --row_count == 0);
    no_jump[3] = no_jump[2] && !(a[1] && --row_count == 0);
    no_jump[4] = no_jump[3] && !(b[1] && --row_count == 0);
    no_jump[5] = no_jump[4] && !(a[2] && --row_count == 0);
    no_jump[6] = no_jump[5] && !(b[2] && --row_count == 0);
    no_jump[7] = no_jump[6] && !(a[3] && --row_count == 0);
	no_jump[8] = no_jump[7] && !(b[3] && --row_count == 0);
	no_jump[9] = no_jump[8] && !(a[4] && --row_count == 0);
	no_jump[10] = no_jump[9] && !(b[4] && --row_count == 0);
	no_jump[11] = no_jump[10] && !(a[5] && --row_count == 0);
	no_jump[12] = no_jump[11] && !(b[5] && --row_count == 0);
	no_jump[13] = no_jump[12] && !(a[6] && --row_count == 0);
	no_jump[14] = no_jump[13] && !(b[6] && --row_count == 0);
	no_jump[15] = no_jump[14] && !(a[7] && --row_count == 0);
	

    //*pg_a1 = svdupq_b32(a[0] && no_jump[0], a[1] && no_jump[2], a[2] && no_jump[4], a[3] && no_jump[6]);
    //*pg_b1 = svdupq_b32(b[0] && no_jump[1], b[1] && no_jump[3], b[2] && no_jump[5], b[3] && no_jump[7]);

	//*pg_a2 = svdupq_b32(a[4] && no_jump[8], a[5] && no_jump[10], a[6] && no_jump[12], a[7] && no_jump[14]);
	//*pg_b2 = svdupq_b32(b[4] && no_jump[9], b[5] && no_jump[11], b[6] && no_jump[13], b[7] && no_jump[15]);

	*pg_a = svzip1_b32(svdupq_b32(a[0] && no_jump[0], a[2] && no_jump[4], a[4] && no_jump[8], a[6] && no_jump[12] ), svdupq_b32(a[1] && no_jump[2], a[3] && no_jump[6], a[5] && no_jump[10], a[7] && no_jump[14]));
	*pg_b = svzip1_b32(svdupq_b32(b[0] && no_jump[1], b[2] && no_jump[5], b[4] && no_jump[9], b[6] && no_jump[13]), svdupq_b32(b[1] && no_jump[3], b[3] && no_jump[7], b[5] && no_jump[11], b[7] && no_jump[15])); 

	aos8_offset(a, stride, t_lo_offset);
	aos8_offset(b, stride, t_hi_offset);
}
