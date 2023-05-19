#pragma once

#include <nnpack/fft-constants.h>
#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>
#include <nnpack/hwinfo.h>

/*
static inline void sve_fft8x8_complex_kernel(
    float tf[restrict static 16 * 4], int channels)
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = 64;
    const int HALF_LENGTH = 32;
    const uint64_t numVals = svcntw()/BLOCK_SIZE; 

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg;
    const svbool_t all = svptrue_b32();
    svfloat32_t b, a, new_b, new_a, new_bt;

    const svuint32_t ind_zip = zip_concat_8(all);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);

    svuint32_t ind_load = indexA(all, (uint32_t []){0,8,1,9,2,10,3,11}, 8, 16);
    const svuint32_t ind_store = indexN(all, 0, 1, 16, 8);
   
    for(int channel =0; channel < channels; channel++){

    for (uint32_t i = 0; i < BLOCK_SIZE/2; i += numVals)
    {

        pg = svwhilelt_b32_s32(i * BLOCK_SIZE, HALF_LENGTH);
    
        a = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2, ind_load);
        b = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2, ind_load);

        // stage1
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_1, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

        // stage2
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_2, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_even, &ind_odd, &ind_zip, &a, &b);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // store
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + 0, ind_store, new_a);
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE, ind_store, new_b);

    }

            tf += LENGTH;
    }
}
*/

//--16--------------------------------------------------------------------------------

//todo add to complex-soa instead of here
inline static void sve_fft16x16_complex_kernel(
    float tf[restrict static 16 * 16], int channels)
{

    const svfloat32_t twiddle_1_r = svzip1(svdupq_f32(COS_0PI_OVER_8, COS_2PI_OVER_8, COS_4PI_OVER_8, COS_6PI_OVER_8), svdupq_f32(COS_1PI_OVER_8, COS_3PI_OVER_8, COS_5PI_OVER_8, COS_7PI_OVER_8));
    const svfloat32_t twiddle_1_i = svzip1(svdupq_f32(SIN_0PI_OVER_8, SIN_2PI_OVER_8, SIN_4PI_OVER_8, SIN_6PI_OVER_8), svdupq_f32(SIN_1PI_OVER_8, SIN_3PI_OVER_8, SIN_5PI_OVER_8, SIN_7PI_OVER_8));
    const svfloat32_t twiddle_1 = svzip1(twiddle_1_r, twiddle_1_i);

    const svfloat32_t twiddle_2 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_3 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);
   
    svbool_t pg;
    const svbool_t all = svptrue_b32();
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int BLOCK_SIZE = 16;
    const int LENGTH = 256;
    const int HALF_LENGTH = 128;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;

    svuint32_t ind_octo_0, ind_octo_1, ind_tetra_0, ind_tetra_1, ind_duo_0, ind_duo_1;
 
    ind_duo(all, &ind_duo_0, &ind_duo_1);
    ind_tetra(all, &ind_tetra_0, &ind_tetra_1);
    ind_octo(all, &ind_octo_0, &ind_octo_1);

    const svuint32_t ind_zip = zip_concat_16(all);

    svuint32_t ind_index = indexN(all, 0, 1, 16*2, 8);
    ind_index = svzip1(ind_index ,svadd_m(all, ind_index, BLOCK_SIZE));

    for(int i =0; i < BLOCK_SIZE/2 * channels; i += numVals){

        pg = svwhilelt_b32_s32(i * BLOCK_SIZE, HALF_LENGTH * channels);
        
        //load
        a = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + 0, ind_index);
        b = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE/2, ind_index);
        
        //stage1
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_1, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_octo_0, &ind_octo_1, &ind_zip, &a, &b);
        
        //stage2
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_2, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_tetra_0, &ind_tetra_1, &ind_zip, &a, &b);

        //stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmulc_twiddle(&pg, &new_b, &twiddle_3, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_duo_0, &ind_duo_1, &ind_zip, &a, &b);

        //stage4
        butterfly(&pg, &a, &b, &new_a, &new_b);

        //store
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + 0, ind_index, new_a);
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE/2, ind_index, new_b);

    }
}
