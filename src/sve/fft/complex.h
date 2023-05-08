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




static inline void sve_ifft8x8_complex_128(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t scaled_twiddle_1_r = svdupq_f32(0.125f * COS_0PI_OVER_4, 0.125f * COS_1PI_OVER_4, 0.125f * COS_2PI_OVER_4, 0.125f * COS_3PI_OVER_4);
    const svfloat32_t scaled_twiddle_1_i = svdupq_f32(0.125f * SIN_0PI_OVER_4, 0.125f * SIN_1PI_OVER_4, 0.125f * SIN_2PI_OVER_4, 0.125f * SIN_3PI_OVER_4);

    const svfloat32_t twiddle_2_r = svdupq_f32(COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2);
    const svfloat32_t twiddle_2_i = svdupq_f32(SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load, pg_vnum_0, pg_vnum_1;
    svfloat32_t ar, ai, br, bi, new_br, new_ar, new_bi, new_ai, tw_br, tw_bi;

              
    const uint64_t numVals = svcntw()/4;

    const svuint32_t ind_zip_interleave = index4(0, 1, 2, 3, 4);
    const svuint32_t ind_zip_concat = index4(0, 2, 1, 3, 4);

    const svuint32_t ind_low = index2(0, 1, 4);
    const svuint32_t ind_high = index2(2, 3, 4);
    
    const svuint32_t ind_load = indexN(svptrue_b32(), 0, 1, 8, 4);
    const svuint32_t ind_store = indexN(svptrue_b32(), 0, 1, 16, 8);


    for (uint32_t i = 0; i < 4; i += numVals)
    {

        pg = svwhilelt_b32_s32(i * 4, 4 * 4);
        
        ar = svld1_gather_index(pg, tf + i * BLOCK_SIZE , ind_load); 
        ai = svld1_gather_index(pg, tf + i * BLOCK_SIZE + LENGTH/2 , ind_load); 
        
        br = svld1_gather_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE / 2, ind_load); 
        bi = svld1_gather_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE / 2 + LENGTH/2, ind_load);
      

         // stage3
        butterfly(&pg, &ar, &br, &new_ar, &new_br);
        butterfly(&pg, &ai, &bi, &new_ai, &new_bi);


        // stage2
        suffle(&pg, &new_ar, &new_br, &ind_low, &ind_high, &ind_zip_interleave, &ar, &br);
        suffle(&pg, &new_ai, &new_bi, &ind_low, &ind_high, &ind_zip_interleave, &ai, &bi);
        mul_twiddle(&pg, &br, &bi, &twiddle_2_r, &twiddle_2_i, &tw_br, &tw_bi);
        butterfly(&pg, &ar, &tw_br, &new_ar, &new_br);
        butterfly(&pg, &ai, &tw_bi, &new_ai, &new_bi);

        // stage1
        suffle(&pg, &new_ar, &new_br, &ind_low, &ind_high, &ind_zip_concat, &ar, &br);
        suffle(&pg, &new_ai, &new_bi, &ind_low, &ind_high, &ind_zip_concat, &ai, &bi);
        mul_twiddle(&pg, &br, &bi, &scaled_twiddle_1_r, &scaled_twiddle_1_i, &tw_br, &tw_bi);
        ar = svmul_m(pg, ar, svdup_f32(0.125f));
        ai = svmul_m(pg, ai, svdup_f32(0.125f));
        butterfly(&pg, &ar, &tw_br, &new_ar, &new_br);
        butterfly(&pg, &ai, &tw_bi, &new_ai, &new_bi);

        // store
        //svst1_scatter_index(pg, tf + i * BLOCK_SIZE + 0, ind_load, new_ar);
        //svst1_scatter_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE/2, ind_load, new_ai);
        //svst1_scatter_index(pg, tf + i * BLOCK_SIZE + LENGTH/2, ind_load, new_br);
        //svst1_scatter_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + LENGTH/2, ind_load, new_bi);

        pg_vnum_0 = svzip1_b32(pg,pg); 
        pg_vnum_1 = svzip2_b32(pg,pg); 
        svst1_vnum(pg_vnum_0, tf + i * BLOCK_SIZE, 0, svzip1(new_ar, new_ai));
        svst1_vnum(pg_vnum_1, tf + i * BLOCK_SIZE, 1, svzip2(new_ar, new_ai));
        svst1_vnum(pg_vnum_0, tf + i * BLOCK_SIZE + LENGTH/2, 0, svzip1(new_br, new_bi));
        svst1_vnum(pg_vnum_1, tf + i * BLOCK_SIZE + LENGTH/2, 1, svzip2(new_br, new_bi));

    }
}



static inline void sve_fft8x8_complex_128(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t twiddle_1_r = svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4);
    const svfloat32_t twiddle_1_i = svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4);

    const svfloat32_t twiddle_2_r = svdupq_f32(COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2);
    const svfloat32_t twiddle_2_i = svdupq_f32(SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load;
    svfloat32_t ar, ai, br, bi, new_br, new_ar, new_bi, new_ai, tw_br, tw_bi;

              
    const uint64_t numVals = svcntw()/4;

    const svuint32_t ind_zip = index4(0, 2, 1, 3, 4);

    const svuint32_t ind_low = index2(0, 1, 4);
    const svuint32_t ind_high = index2(2, 3, 4);
    const svuint32_t ind_even = index2(0, 2, 4);
    const svuint32_t ind_odd = index2(1, 3, 4);

    
    const svuint32_t ind_load = indexN(svptrue_b32(), 0, 1, 16, 4);


    for (uint32_t i = 0; i < 4; i += numVals)
    {

        pg = svwhilelt_b32_s32(i * 4, 4 * 4);
        
        ar = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 , ind_load); 
        ai = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE , ind_load); 
        
        br = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2, ind_load); 
        bi = svld1_gather_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2 + BLOCK_SIZE, ind_load);
      

        // stage1
        butterfly(&pg, &ar, &br, &new_ar, &new_br);
        butterfly(&pg, &ai, &bi, &new_ai, &new_bi);
        mulc_twiddle(&pg, &new_br, &new_bi, &twiddle_1_r, &twiddle_1_i, &tw_br, &tw_bi);
        suffle(&pg, &new_ar, &tw_br, &ind_low, &ind_high, &ind_zip, &ar, &br);
        suffle(&pg, &new_ai, &tw_bi, &ind_low, &ind_high, &ind_zip, &ai, &bi);

        // stage2
        butterfly(&pg, &ar, &br, &new_ar, &new_br);
        butterfly(&pg, &ai, &bi, &new_ai, &new_bi);
        mulc_twiddle(&pg, &new_br, &new_bi, &twiddle_2_r, &twiddle_2_i, &tw_br, &tw_bi);
        suffle(&pg, &new_ar, &tw_br, &ind_even, &ind_odd, &ind_zip, &ar, &br);
        suffle(&pg, &new_ai, &tw_bi, &ind_even, &ind_odd, &ind_zip, &ai, &bi);

        // stage3
        butterfly(&pg, &ar, &br, &new_ar, &new_br);
        butterfly(&pg, &ai, &bi, &new_ai, &new_bi);

        // store
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + 0, ind_load, new_ar);
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE/2, ind_load, new_ai);
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE, ind_load, new_br);      
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE + BLOCK_SIZE/2, ind_load, new_bi);

    }
}



//                      _256
static inline void sve_fft8x8_complex(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width;
    const int dim = 2;                       // complex number
    const uint64_t numVals = svcntw(); // a and b

    const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);

    const int to_bytes = 4;
    const svuint32_t ind_load = index8(to_bytes * 0, to_bytes * 8, to_bytes * 1, to_bytes * 9, to_bytes * 2, to_bytes * 10, to_bytes * 3, to_bytes * 11, to_bytes * 16);
    const svuint32_t ind_store = index8(to_bytes * 0,to_bytes *1,to_bytes *2,to_bytes *3,to_bytes *4,to_bytes *5,to_bytes *6,to_bytes *7,to_bytes *16); 

    for (uint32_t i = 0; i < LENGTH/dim; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, LENGTH / dim);
    
        a = svld1_gather_offset(pg, tf + i * 2, ind_load);
        b = svld1_gather_offset(pg, tf + i * 2 + BLOCK_SIZE / 2, ind_load);

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
        svst1_scatter_offset(pg, tf + i * 2 + 0, ind_store, new_a);
        svst1_scatter_offset(pg, tf + i * 2 + BLOCK_SIZE, ind_store, new_b);

    }
}

static inline void sve_ifft8x8_complex(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t scaled_twiddle_1 = svzip1(svdupq_f32(0.125f * COS_0PI_OVER_4, 0.125f * COS_1PI_OVER_4, 0.125f * COS_2PI_OVER_4, 0.125f * COS_3PI_OVER_4), svdupq_f32(0.125f * SIN_0PI_OVER_4, 0.125f * SIN_1PI_OVER_4, 0.125f * SIN_2PI_OVER_4, 0.125f * SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width;
    
    const int dim = 2;                       // complex number
    const uint64_t numVals = svcntw();

    const svuint32_t ind_zip_interleave = index8(0, 2, 1, 3, 4, 6, 5, 7, 8);
    const svuint32_t ind_zip_concat = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    
    const int to_bytes = 4;
    const svuint32_t offsets = index8(to_bytes * 0, to_bytes * 32, to_bytes * 1, to_bytes * 33, to_bytes * 2, to_bytes * 34, to_bytes * 3, to_bytes * 35, to_bytes * 8);
    //const svuint32_t offsets = index8(to_bytes * 0, to_bytes * 8, to_bytes * 1, to_bytes * 9, to_bytes * 2, to_bytes * 10, to_bytes * 3, to_bytes * 11, to_bytes * 16);

    for (uint32_t i = 0; i < LENGTH/dim; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, LENGTH / dim);

        a = svld1_gather_offset(pg, tf + i, offsets);
        b = svld1_gather_offset(pg, tf + i + BLOCK_SIZE/2, offsets);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // stage2
        suffle(&pg, &new_a, &new_b, &ind_low, &ind_high, &ind_zip_interleave, &a, &b);
        cmul_twiddle(&pg, &b, &twiddle_2, &new_bt);
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // stage1
        suffle(&pg, &new_a, &new_b, &ind_low, &ind_high, &ind_zip_concat, &a, &b);
        cmul_twiddle(&pg, &b, &scaled_twiddle_1, &new_bt);
        a = svmul_m(pg, a, svdup_f32(0.125f));
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // store
        svst1(pg, tf + i, new_a);
        svst1(pg, tf + i + LENGTH/2, new_b);
    }
}


//--16--------------------------------------------------------------------------------


inline static void sve_fft16_complex_512(
    float tf[restrict static 16 * 16])
{

    const svfloat32_t twiddle_1_r = svzip1(svdupq_f32(COS_0PI_OVER_8, COS_2PI_OVER_8, COS_4PI_OVER_8, COS_6PI_OVER_8), svdupq_f32(COS_1PI_OVER_8, COS_3PI_OVER_8, COS_5PI_OVER_8, COS_7PI_OVER_8));
    const svfloat32_t twiddle_1_i = svzip1(svdupq_f32(SIN_0PI_OVER_8, SIN_2PI_OVER_8, SIN_4PI_OVER_8, SIN_6PI_OVER_8), svdupq_f32(SIN_1PI_OVER_8, SIN_3PI_OVER_8, SIN_5PI_OVER_8, SIN_7PI_OVER_8));
    const svfloat32_t twiddle_1 = svzip1(twiddle_1_r, twiddle_1_i);

    const svfloat32_t twiddle_2 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_3 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);
   
    svbool_t pg;
    const svbool_t all = svptrue_b32();
    svfloat32_t b, a, new_b, new_a, new_bt;

    int BLOCK_SIZE = 16;
    int LENGTH = 256;
    int HALF_LENGTH = 128;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;


    const svuint32_t ind_octo_0 = indexN(all,0, 1, 16, 8);
    const svuint32_t ind_octo_1 = svadd_m(all,ind_octo_0, 8);

    svuint32_t quad = indexN(all, 0, 8, 16, 2);
    quad = svzip1(quad, svadd_m(all, quad, 2));
    const svuint32_t ind_tetra_0 = svzip1(quad, svadd_m(all, quad, 1));
    const svuint32_t ind_tetra_1 = svadd_m(all, ind_tetra_0, 4);

    const svuint32_t even = indexN(all, 0, 4, 16, 4);
    const svuint32_t ind_duo_0 = svzip1(even, svadd_m(all, even, 1));
    const svuint32_t ind_duo_1 = svadd_m(all, ind_duo_0, 2);

    const svuint32_t ind_zip = zip_concat_16(all);

    svuint32_t ind_index = indexN(all, 0, 1, 16*2, 8);
    ind_index = svzip1(ind_index ,svadd_m(all, ind_index, BLOCK_SIZE));

    for(int i =0; i < BLOCK_SIZE/2; i += numVals){

        pg = svwhilelt_b32_s32(i * BLOCK_SIZE, HALF_LENGTH);
        
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


static inline void sve_ifft16x16_complex(
    float tf[restrict static 16 * 4])
{

    const svfloat32_t scaled_twiddle_1_r = svzip1(svdupq_f32(COS_0PI_OVER_8 * 0.0625f, COS_2PI_OVER_8 * 0.0625f, COS_4PI_OVER_8 * 0.0625f, COS_6PI_OVER_8 * 0.0625f), svdupq_f32(COS_1PI_OVER_8 * 0.0625f, COS_3PI_OVER_8 * 0.0625f, COS_5PI_OVER_8 * 0.0625f, COS_7PI_OVER_8 * 0.0625f));
    const svfloat32_t scaled_twiddle_1_i = svzip1(svdupq_f32(SIN_0PI_OVER_8 * 0.0625f, SIN_2PI_OVER_8 * 0.0625f, SIN_4PI_OVER_8 * 0.0625f, SIN_6PI_OVER_8 * 0.0625f), svdupq_f32(SIN_1PI_OVER_8 * 0.0625f, SIN_3PI_OVER_8 * 0.0625f, SIN_5PI_OVER_8 * 0.0625f, SIN_7PI_OVER_8 * 0.0625f));
    const svfloat32_t scaled_twiddle_1 = svzip1(scaled_twiddle_1_r, scaled_twiddle_1_i);

    const svfloat32_t twiddle_2 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_3 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);
   
    svbool_t pg;
    const svbool_t all = svptrue_b32();
    svfloat32_t b, a, new_b, new_a, new_bt;

    int BLOCK_SIZE = 16;
    int LENGTH = 256;
    int HALF_LENGTH = 128;
    const uint64_t numVals = svcntw()/BLOCK_SIZE;


    const svuint32_t ind_octo_0 = indexN(all,0, 1, 16, 8);
    const svuint32_t ind_octo_1 = svadd_m(all,ind_octo_0, 8);

    svuint32_t quad = indexN(all, 0, 8, 16, 2);
    quad = svzip1(quad, svadd_m(all, quad, 2));
    const svuint32_t ind_tetra_0 = svzip1(quad, svadd_m(all, quad, 1));
    const svuint32_t ind_tetra_1 = svadd_m(all, ind_tetra_0, 4);


    const svuint32_t ind_zip_concat = zip_concat_16(all);

    svuint32_t interleave = indexN(all, 0, 4, 16, 4);
    interleave = svzip1(interleave, svadd_m(all, interleave, 1));
    const svuint32_t ind_zip_interleave = svzip1(interleave, svadd_m(all, interleave, 2));
    
    svuint32_t ind_zip_mix = indexN(all, 0, 8, 16, 2);
    ind_zip_mix = svzip1(ind_zip_mix, svadd_m(all, ind_zip_mix, 1));
    ind_zip_mix = svzip1(ind_zip_mix, svadd_m(all, ind_zip_mix, 4));
    ind_zip_mix = svzip1(ind_zip_mix, svadd_m(all, ind_zip_mix, 2));

    svuint32_t ind_index = indexN(all, 0, 1, 16, 8);
    ind_index = svzip1(ind_index ,svadd_m(all, ind_index, HALF_LENGTH));

    for(int i =0; i < BLOCK_SIZE/2; i += numVals)
    {

        pg = svwhilelt_b32_s32(i * BLOCK_SIZE, HALF_LENGTH);

        a = svld1_gather_index(pg, tf + i * BLOCK_SIZE + 0, ind_index);
        b = svld1_gather_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE/2, ind_index);

        // stage4
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // stage3
        suffle(&pg, &new_a, &new_b, &ind_octo_0, &ind_octo_1, &ind_zip_interleave, &a, &b);
        cmul_twiddle(&pg, &b, &twiddle_3, &new_bt);
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // stage2
        suffle(&pg, &new_a, &new_b, &ind_octo_0, &ind_octo_1, &ind_zip_mix, &a, &b);
        cmul_twiddle(&pg, &b, &twiddle_2, &new_bt);
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // stage1
        suffle(&pg, &new_a, &new_b, &ind_octo_0, &ind_octo_1, &ind_zip_concat, &a, &b);
        cmul_twiddle(&pg, &b, &scaled_twiddle_1, &new_bt);
        a = svmul_m(pg, a, svdup_f32(0.0625f));
        butterfly(&pg, &a, &new_bt, &new_a, &new_b);

        // store
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE + 0, ind_index, new_a);
        svst1_scatter_index(pg, tf + i * BLOCK_SIZE + BLOCK_SIZE/2, ind_index, new_b);
        
    }
}