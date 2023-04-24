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
#include <sve/fft/soa.h>
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

    svbool_t pg, pg_load;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width;
    const int dim = 2;                       // complex number
    const uint64_t numVals = svcntw() * dim; // a and b

    const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);

    const int to_bytes = 4;
    const svuint32_t ind_load = index8(to_bytes * 0, to_bytes * 8, to_bytes * 1, to_bytes * 9, to_bytes * 2, to_bytes * 10, to_bytes * 3, to_bytes * 11, to_bytes * 16);
    const svuint32_t ind_store = index8(to_bytes * 0,to_bytes *1,to_bytes *2,to_bytes *3,to_bytes *4,to_bytes *5,to_bytes *6,to_bytes *7,to_bytes *16); 

    for (uint32_t i = 0; i < LENGTH; i += numVals)
    {

        pg = svwhilelt_b32_s32(i/dim, LENGTH / dim);
        pg_load = svzip1_b32(pg, pg);

        a = svld1_gather_offset(pg_load, tf + i, ind_load);
        b = svld1_gather_offset(pg_load, tf + i + BLOCK_SIZE / 2, ind_load);

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
        svst1_scatter_offset(pg_load, tf + i  + 0, ind_store, new_a);
        svst1_scatter_offset(pg_load, tf + i  + BLOCK_SIZE, ind_store, new_b);

    }
}

static inline void sve_ifft8x8_complex(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t scaled_twiddle_1 = svzip1(svdupq_f32(0.125f * COS_0PI_OVER_4, 0.125f * COS_1PI_OVER_4, 0.125f * COS_2PI_OVER_4, 0.125f * COS_3PI_OVER_4), svdupq_f32(0.125f * SIN_0PI_OVER_4, 0.125f * SIN_1PI_OVER_4, 0.125f * SIN_2PI_OVER_4, 0.125f * SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width;
    
    const int dim = 2;                       // complex number
    const uint64_t numVals = svcntw() * dim; // a and b

    const svuint32_t ind_zip_interleave = index8(0, 2, 1, 3, 4, 6, 5, 7, 8);
    const svuint32_t ind_zip_concat = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    
    const int to_bytes = 4;
    const svuint32_t offsets = index8(to_bytes * 0, to_bytes * 32, to_bytes * 1, to_bytes * 33, to_bytes * 2, to_bytes * 34, to_bytes * 3, to_bytes * 35, to_bytes * 8);
    //const svuint32_t offsets = index8(to_bytes * 0, to_bytes * 8, to_bytes * 1, to_bytes * 9, to_bytes * 2, to_bytes * 10, to_bytes * 3, to_bytes * 11, to_bytes * 16);

    for (uint32_t i = 0; i < LENGTH; i += numVals)
    {

        pg = svwhilelt_b32_s32(i / dim, LENGTH / dim);
        pg_load = svzip1_b32(pg, pg); // svwhilelt_b32_s32(i, LENGTH);

        a = svld1_gather_offset(pg_load, tf + i/2, offsets);
        b = svld1_gather_offset(pg_load, tf + i/2 + BLOCK_SIZE/2, offsets);

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
        svst1(pg_load, tf + i/2, new_a);
        svst1(pg_load, tf + i/2 + LENGTH/2, new_b);
    }
}

static inline void fft8_armComputeLib(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    // const float f8t0[16] = {1.0f,0.0f,  1.0f,  0.0f,                 1.0f,  0.0f,  1.0f,  0.0f,                  1.0f,0.0f,  1.0f,  0.0f,                   1.0f, 0.0f, 1.0f, 0.0f};
    const svfloat32_t f0t = svdupq_f32(1.0f, 0.0f, 1.0f, 0.0f);
    const float f8t1[16] = {1.0f, 0.0f, SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f, 0.0f, -SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, 1.0f, SQRT2_OVER_2, SQRT2_OVER_2};
    const float f8t2[16] = {1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f};
    const float f8t3[16] = {1.0f, 0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, 1.0f, SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f, 0.0f, SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, -1.0f, -SQRT2_OVER_2, SQRT2_OVER_2};
    const float f8t4[16] = {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f};
    const float f8t5[16] = {1.0f, 0.0f, -SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, -1.0f, SQRT2_OVER_2, SQRT2_OVER_2, -1.0f, 0.0f, SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, 1.0f, -SQRT2_OVER_2, -SQRT2_OVER_2};
    const float f8t6[16] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f};
    const float f8t7[16] = {1.0f, 0.0f, SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, 1.0f, -SQRT2_OVER_2, SQRT2_OVER_2, -1.0f, 0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f, SQRT2_OVER_2, -SQRT2_OVER_2};

    svfloat32_t f0, f1, f2, f3, f4, f5, f6, f7;
    f0 = svdup_f32(0.0f);
    f1 = svdup_f32(0.0f);
    f2 = svdup_f32(0.0f);
    f3 = svdup_f32(0.0f);
    f4 = svdup_f32(0.0f);
    f5 = svdup_f32(0.0f);
    f6 = svdup_f32(0.0f);
    f7 = svdup_f32(0.0f);
    svbool_t pg;
    const svbool_t all_active = svptrue_b32();
    const svbool_t r_active = svdupq_b32(1, 0, 1, 0);
    const svbool_t i_active = svdupq_b32(0, 1, 0, 1);

    uint64_t numVals = svlen(f0);

    for (uint32_t i = 0; i < 16; i += numVals)
    {
        pg = svwhilelt_b32_s32(i, 16);

        const svfloat32_t fd = svzip1(svld1(pg, t + i / 2), svld1(pg, t + i / 2 + 8));

        f0 = svcmla_m(pg, f0, f0t, fd, 0);
        f0 = svcmla_m(pg, f0, f0t, fd, 90);

        const svfloat32_t f1t = svld1(pg, f8t1 + i);
        f1 = svcmla_m(pg, f1, f1t, fd, 0);
        f1 = svcmla_m(pg, f1, f1t, fd, 90);

        const svfloat32_t f2t = svld1(pg, f8t2 + i);
        f2 = svcmla_m(pg, f2, f2t, fd, 0);
        f2 = svcmla_m(pg, f2, f2t, fd, 90);

        const svfloat32_t f3t = svld1(pg, f8t3 + i);
        f3 = svcmla_m(pg, f3, f3t, fd, 0);
        f3 = svcmla_m(pg, f3, f3t, fd, 90);

        const svfloat32_t f4t = svld1(pg, f8t4 + i);
        f4 = svcmla_m(pg, f4, f4t, fd, 0);
        f4 = svcmla_m(pg, f4, f4t, fd, 90);

        const svfloat32_t f5t = svld1(pg, f8t5 + i);
        f5 = svcmla_m(pg, f5, f5t, fd, 0);
        f5 = svcmla_m(pg, f5, f5t, fd, 90);

        const svfloat32_t f6t = svld1(pg, f8t6 + i);
        f6 = svcmla_m(pg, f6, f6t, fd, 0);
        f6 = svcmla_m(pg, f6, f6t, fd, 90);

        const svfloat32_t f7t = svld1(pg, f8t7 + i);
        f7 = svcmla_m(pg, f7, f7t, fd, 0);
        f7 = svcmla_m(pg, f7, f7t, fd, 90);
    }

    f[0] = svaddv(r_active, f0);
    f[1] = svaddv(i_active, f0);

    f[2] = svaddv(r_active, f1);
    f[3] = svaddv(i_active, f1);

    f[4] = svaddv(r_active, f2);
    f[5] = svaddv(i_active, f2);

    f[6] = svaddv(r_active, f3);
    f[7] = svaddv(i_active, f3);

    f[8] = svaddv(r_active, f4);
    f[9] = svaddv(i_active, f4);

    f[10] = svaddv(r_active, f5);
    f[11] = svaddv(i_active, f5);

    f[12] = svaddv(r_active, f6);
    f[13] = svaddv(i_active, f6);

    f[14] = svaddv(r_active, f7);
    f[15] = svaddv(i_active, f7);
}

inline static void fft16(
    const float t[restrict static 32],
    float f[restrict static 32])
{
}