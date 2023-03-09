#pragma once

#include <nnpack/fft-constants.h>
#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

static inline void svprint_f(svbool_t pg, svfloat32_t printme, const int n)
{

    float32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%f ", tmp[i]);

    printf("\n");
}

static inline void svprint_i(svbool_t pg, svint32_t printme, const int n)
{

    int32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%d ", tmp[i]);

    printf("\n");
}

static inline void print_array_cf(const float *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%f ", arr[i]);

    printf("\n");
}

static inline void print_array_f(float *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%f ", arr[i]);

    printf("\n");
}

inline static void fft4(
    const float t[restrict static 8],
    float f[restrict static 8])
{
}

static inline void fft8_empty(
    const float t[restrict static 16],
    float f[restrict static 16])
{
}

static inline void fft8(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    float twiddle_factors[16] = {COS_0PI_OVER_4, SIN_0PI_OVER_4, COS_1PI_OVER_4, SIN_1PI_OVER_4, COS_2PI_OVER_4, SIN_2PI_OVER_4, COS_3PI_OVER_4, SIN_3PI_OVER_4,
                                 COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2, COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2};

    svbool_t pg,pg_load;

    svfloat32_t b, a, new_b, new_a, twiddle, new_bt;
    svuint32_t ind_a,ind_b;

    const uint64_t numVals = svcntw();
   
    uint32_t index_zip [8] = {0,2,4,6,1,3,5,7};
    uint32_t index_low [4] = {0,1,2,3};
    uint32_t index_high[4] = {4,5,6,7};
    uint32_t index_even[4] = {0,1,4,5};
    uint32_t index_odd [4] = {2,3,6,7};

    new_bt = svdup_f32(0.0f);
    //r: bbbb aaaa i: bbbb aaaa
    // bbbb bbbb aaaa aaaa
    //R 0,1,2,3 4,5,6,7
    //I 0,1,2,3 4,5,6,7
    
    //numvals >= 8
    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);
        pg_load = svwhilelt_b32_s32(i/2, 4);

        // svld1_vnum
        a = svzip1(svld1(pg_load, t + i/2),svld1(pg_load, t+ i/2 +8));       
        b = svzip1(svld1(pg_load, t + i/2 +4),svld1(pg_load, t+ i/2 +8 +4));

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        twiddle = svld1(pg, twiddle_factors + i);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);


        const svuint32_t ind_zip = svld1(pg, index_zip + i);

        ind_a = svld1(pg, index_low + i);
        ind_b = svld1(pg, index_high + i);
        a = svtbl(svzip1(svtbl(new_a,ind_a),svtbl(new_bt,ind_a)), ind_zip);
        b = svtbl(svzip1(svtbl(new_a,ind_b),svtbl(new_bt,ind_b)), ind_zip);
        new_bt = svdup_f32(0.0f); 


        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        twiddle = svld1(pg, twiddle_factors + i +8);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);

        
        ind_a = svld1(pg, index_even + i);
        ind_b = svld1(pg, index_odd + i);
        a = svtbl(svzip1(svtbl(new_a,ind_a),svtbl(new_bt,ind_a)), ind_zip);
        b = svtbl(svzip1(svtbl(new_a,ind_b),svtbl(new_bt,ind_b)), ind_zip);

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);


        svst1(pg, f + i + 8, new_b);
        svst1(pg, f + i, new_a);
       
    }

}

static inline void fft8_old(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    float twiddle_factors[16] = {COS_0PI_OVER_4, SIN_0PI_OVER_4, COS_1PI_OVER_4, SIN_1PI_OVER_4, COS_2PI_OVER_4, SIN_2PI_OVER_4, COS_3PI_OVER_4, SIN_3PI_OVER_4,
                                 COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2, COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2};

    svbool_t pg;

    svfloat32_t b, a, new_b, new_a, twiddle, new_bt;
    svint32_t ind, ind_out;

    const uint64_t numVals = svcntw();
    float tmp[16] = {0.0f};

    int offset0[8] = {0, 8, 1, 9, 2, 10, 3, 11};
    int offset1[8] = {8, 9, 10, 11, 0, 1, 2, 3};
    int offset2[8] = {8, 9, 12, 13, 0, 1, 4, 5};

    new_bt = svdup_f32(0.0f);

    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);

        ind = svld1(pg, offset0 + i);
        a = svld1_gather_index(pg, t, ind);
        b = svld1_gather_index(pg, t + 4, ind);

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        twiddle = svld1(pg, twiddle_factors + i);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);

        svst1(pg, f + i, new_bt);
        svst1(pg, f + i + 8, new_a);
    }

    new_bt = svdup_f32(0.0f);

    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);

        ind = svld1(pg, offset1 + i);
        a = svld1_gather_index(pg, f, ind);
        b = svld1_gather_index(pg, f + 4, ind);

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        twiddle = svld1(pg, twiddle_factors + i + 8);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);

        svst1(pg, tmp + i, new_bt);
        svst1(pg, tmp + i + 8, new_a);
    }

    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);

        ind = svld1(pg, offset2 + i);
        a = svld1_gather_index(pg, tmp, ind);
        b = svld1_gather_index(pg, tmp + 2, ind);

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        svst1(pg, f + i + 8, new_b);
        svst1(pg, f + i, new_a);
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