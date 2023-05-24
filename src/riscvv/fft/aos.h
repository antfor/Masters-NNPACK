#pragma once

#include <stddef.h>
#include <stdint.h>

#include <nnpack/fft-constants.h>
#include <riscvv/fft/fft-util.h>

inline static __epi_2xi32 aos4_offset_r(const __uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

    // Index 2
    return indexA((uint32_t []){0, cSum_n[1] * stride * 4}, 2, 4, gvl);
}

inline static __epi_2xi32 aos4_offset_i(const __uint32_t *n, size_t stride, long gvl)
{
    int cSum_n[3];
    cSum_n[0] = n[0];
    cSum_n[1] = cSum_n[0] + n[1];
    cSum_n[2] = cSum_n[1] + n[2];

	// Index 2
    return indexA((uint32_t []){cSum_n[0] * stride * 4, cSum_n[2] * stride * 4}, 2, 4, gvl);
}

inline static __epi_2xi1 aos4_mask_a_r(__uint32_t *no_jump, __uint32_t *a, __uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((__uint32_t[]){
                a[0] && no_jump[0],
                a[2] && no_jump[4],
                a[0] && no_jump[0],
                a[2] && no_jump[4]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_a_i(__uint32_t *no_jump, __uint32_t *a, __uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((__uint32_t[]){
                a[1] && no_jump[2],
                a[3] && no_jump[6],
                a[1] && no_jump[2],
                a[3] && no_jump[6]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_b_r(__uint32_t *no_jump, __uint32_t *a, __uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((__uint32_t[]){
                b[0] && no_jump[1],
                b[2] && no_jump[5],
                b[0] && no_jump[1],
                b[2] && no_jump[5]
                },gvl));
}

inline static __epi_2xi1 aos4_mask_b_i(__uint32_t *no_jump, __uint32_t *a, __uint32_t *b, long gvl) {
    return __builtin_epi_cast_2xi1_2xi32(dupq_i((__uint32_t[]){
                b[1] && no_jump[3],
                b[3] && no_jump[7],
                b[1] && no_jump[3],
                b[3] && no_jump[7]
                },gvl));
}

static void jump_arr(__uint32_t *no_jump, __uint32_t *a, __uint32_t *b, uint32_t row_count) {
    no_jump[0] = 1;
    no_jump[1] = no_jump[0] && !(a[0] && --row_count == 0);
    no_jump[2] = no_jump[1] && !(b[0] && --row_count == 0);
    no_jump[3] = no_jump[2] && !(a[1] && --row_count == 0);
    no_jump[4] = no_jump[3] && !(b[1] && --row_count == 0);
    no_jump[5] = no_jump[4] && !(a[2] && --row_count == 0);
    no_jump[6] = no_jump[5] && !(b[2] && --row_count == 0);
    no_jump[7] = no_jump[6] && !(a[3] && --row_count == 0);
}