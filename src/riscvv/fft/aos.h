#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <riscvv/fft/fft-util.h>


//---4x4-------------------------------------------------------------

inline static __epi_2xi32 aos4_offset_r(uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

    // Index 2
    return indexA((uint32_t []){0, cSum_n[1] * stride * 4}, 2, 4, gvl);
}

inline static __epi_2xi32 aos4_offset_i(uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

	// Index 2
    return indexA((uint32_t []){cSum_n[0] * stride * 4, cSum_n[2] * stride * 4}, 2, 4, gvl);
}

inline static __epi_2xi1 aos4_mask_a_r(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){
                a[0] && no_jump[0],
                a[2] && no_jump[4],
                a[0] && no_jump[0],
                a[2] && no_jump[4]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_a_i(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){
                a[1] && no_jump[2],
                a[3] && no_jump[6],
                a[1] && no_jump[2],
                a[3] && no_jump[6]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_b_r(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){
                b[0] && no_jump[1],
                b[2] && no_jump[5],
                b[0] && no_jump[1],
                b[2] && no_jump[5]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_b_i(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){
                b[1] && no_jump[3],
                b[3] && no_jump[7],
                b[1] && no_jump[3],
                b[3] && no_jump[7]
                },gvl));
}

static void jump_arr4(uint32_t *no_jump, uint32_t *a, uint32_t *b, uint32_t row_count) {
    no_jump[0] = 1;
    no_jump[1] = no_jump[0] && !(a[0] && --row_count == 0);
    no_jump[2] = no_jump[1] && !(b[0] && --row_count == 0);
    no_jump[3] = no_jump[2] && !(a[1] && --row_count == 0);
    no_jump[4] = no_jump[3] && !(b[1] && --row_count == 0);
    no_jump[5] = no_jump[4] && !(a[2] && --row_count == 0);
    no_jump[6] = no_jump[5] && !(b[2] && --row_count == 0);
    no_jump[7] = no_jump[6] && !(a[3] && --row_count == 0);
}

//---8x8-------------------------------------------------------------



inline static __epi_2xi32 aos8_offset_r(uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[7];
	cSum_n[0] = n[0];
	cSum_n[1] = cSum_n[0] + n[1];
	cSum_n[2] = cSum_n[1] + n[2];
	cSum_n[3] = cSum_n[2] + n[3];
	cSum_n[4] = cSum_n[3] + n[4];
	cSum_n[5] = cSum_n[4] + n[5];
	cSum_n[6] = cSum_n[5] + n[6];

    // Index 2
    return indexA((uint32_t []){0, cSum_n[1] * stride * 4, cSum_n[3] * stride * 4, cSum_n[5] * stride * 4}, 4, 4, gvl);
}

inline static __epi_2xi32 aos8_offset_i(uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[7];
	cSum_n[0] = n[0];
	cSum_n[1] = cSum_n[0] + n[1];
	cSum_n[2] = cSum_n[1] + n[2];
	cSum_n[3] = cSum_n[2] + n[3];
	cSum_n[4] = cSum_n[3] + n[4];
	cSum_n[5] = cSum_n[4] + n[5];
	cSum_n[6] = cSum_n[5] + n[6];

	// Index 2
    return indexA((uint32_t []){cSum_n[0] * stride * 4, cSum_n[2] * stride * 4, cSum_n[4] * stride * 4, cSum_n[6] * stride * 4}, 4, 4, gvl);
}


inline static __epi_2xi1 aos8_mask_a_r(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){a[0] && no_jump[0], a[2] && no_jump[4], a[4] && no_jump[8], a[6] && no_jump[12]},gvl));
}

inline static __epi_2xi1 aos8_mask_a_i(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){a[1] && no_jump[2], a[3] && no_jump[6], a[5] && no_jump[10], a[7] && no_jump[14]},gvl));
}

inline static __epi_2xi1 aos8_mask_b_r(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){b[0] && no_jump[1], b[2] && no_jump[5], b[4] && no_jump[9], b[6] && no_jump[13]},gvl));
}

inline static __epi_2xi1 aos8_mask_b_i(uint32_t *no_jump, uint32_t *a, uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((uint32_t[]){b[1] && no_jump[3], b[3] && no_jump[7], b[5] && no_jump[11], b[7] && no_jump[15]},gvl));
}

static void jump_arr8(uint32_t *no_jump, uint32_t *a, uint32_t *b, uint32_t row_count) {
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
}