#pragma once

// #include <nnpack/fft-constants.h>
#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

///*
#define SQRT2_OVER_2 0x1.6A09E6p-1f

#define COS_0PI_OVER_4 1.0f
#define COS_1PI_OVER_4 SQRT2_OVER_2
#define COS_2PI_OVER_4 0.0f
#define COS_3PI_OVER_4 -SQRT2_OVER_2

#define COS_0PI_OVER_2 1.0f
#define COS_1PI_OVER_2 0.0f

#define SIN_0PI_OVER_4 0.0f
#define SIN_1PI_OVER_4 SQRT2_OVER_2
#define SIN_2PI_OVER_4 1.0f
#define SIN_3PI_OVER_4 SQRT2_OVER_2

#define SIN_0PI_OVER_2 0.0f
#define SIN_1PI_OVER_2 1.0f
// */

static inline void svprint_f(svbool_t pg, svfloat32_t printme, const int n)
{

    float32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%f ", tmp[i]);

    printf("\n");
}

static inline void svprintf_f32(svbool_t pg, svfloat32_t printme, const int n, const int m)
{

    float32_t tmp[n];

    svst1(pg, tmp, printme);
    printf("svfloat start");
    for (int i = 0; i < n; i += m)
    {
        printf("\n    ");
        for (int j = 0; j < m; j++)
            printf("%f ", tmp[i * m + j]);
    }
    printf("\n svfloat end \n");
}

static inline void svprint_i(svbool_t pg, svint32_t printme, const int n)
{

    int32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%d ", tmp[i]);

    printf("\n");
}

static inline void svprint_ui(svbool_t pg, svuint32_t printme, const int n)
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
//
static inline void fft8_min8(
    const float t[restrict static 16],
    float f[restrict static 16])
{
    svfloat32_t a, b;
    svbool_t pg;
    const uint64_t numVals = svcntw();

    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);
        a = svld1(pg, t + i);
        b = svld1(pg, t + i + 8);

        svst1(pg, f + i, a);
        svst1(pg, f + i + 8, b);
    }
}
//
static inline void fft8_min16(
    const float t[restrict static 16],
    float f[restrict static 16])
{
    svfloat32_t ba, ab;
    svbool_t pg;
    const uint64_t numVals = svcntw();

    for (uint32_t i = 0; i < 16; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 16);
        ba = svld1(pg, t + i);

        svst1(pg, f + i, ba);
    }
}

static inline void butterfly(svbool_t *pg, svfloat32_t *a, svfloat32_t *b, svfloat32_t *new_a, svfloat32_t *new_b)
{
    *new_a = svadd_m(*pg, *a, *b);
    *new_b = svsub_m(*pg, *a, *b);
}

static inline void mul_twiddle(svbool_t *pg, svfloat32_t *br, svfloat32_t *bi, const svfloat32_t *tr, const svfloat32_t *ti, svfloat32_t *new_br, svfloat32_t *new_bi)
{
    *new_br = svadd_m(*pg, svmul_m(*pg, *tr, *br), svmul_m(*pg, *ti, *bi));
    *new_bi = svsub_m(*pg, svmul_m(*pg, *tr, *bi), svmul_m(*pg, *ti, *br));
}

static inline void cmul_twiddle(svbool_t *pg, svfloat32_t *b, const svfloat32_t *t, svfloat32_t *new_b)
{
    *new_b = svdup_f32(0.0f);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 0);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 270);
}

static inline void suffle(svbool_t *pg, svfloat32_t *a, svfloat32_t *b, const svuint32_t *ind_a, const svuint32_t *ind_b, const svuint32_t *ind_zip, svfloat32_t *new_a, svfloat32_t *new_b)
{

    *new_a = svtbl(svzip1(svtbl(*a, *ind_a), svtbl(*b, *ind_a)), *ind_zip);
    *new_b = svtbl(svzip1(svtbl(*a, *ind_b), svtbl(*b, *ind_b)), *ind_zip);
}

static inline svuint32_t index2(uint32_t a, uint32_t b, uint32_t step)
{
    return svzip1(svindex_u32(a, step), svindex_u32(b, step));
}

static inline svuint32_t index4(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t step)
{
    // return svzip1(svzip1(svindex_u32(a, step), svindex_u32(c, step)), svzip1(svindex_u32(b, step), svindex_u32(d, step)));
    return svzip1(index2(a, c, step), index2(b, d, step));
}

static inline svuint32_t index8(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7, uint32_t step)
{
    // 0426 1537
    // return svzip1(svzip1(svzip1(svindex_u32(i0, step), svindex_u32(i4, step)), svzip1(svindex_u32(i2, step), svindex_u32(i6, step))),  svzip1(svzip1(svindex_u32(i1, step), svindex_u32(i5, step)), svzip1(svindex_u32(i3, step), svindex_u32(i7, step))));
    return svzip1(index4(i0, i2, i4, i6, step), index4(i1, i3, i5, i7, step));
}

//                      _256
static inline void fft8x8(
    const float t[restrict static 16 * 4],
    float f[restrict static 16 * 4],
    size_t f_stride)
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load;
    svfloat32_t b, a, new_b, new_a, twiddle, new_bt;

    const uint64_t numVals = svcntw() * 2; // a and b

    const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);
    const svuint32_t ind_load = index8(0, 8, 1, 9, 2, 10, 3, 11, 16);
    const svuint32_t ind_store = index8(0, 1, 2, 3, 4, 5, 6, 7, 16);

    
    const svuint32_t offsets = index8(f_stride * 0 + 0, f_stride * 0 + 4, f_stride * 1 + 0, f_stride * 1 + 4, f_stride * 2 + 0, f_stride * 2 + 4, f_stride * 3 + 0, f_stride * 3 + 4  ,f_stride * BLOCK_SIZE);
    // printf("start\n");

    for (uint32_t i = 0; i < LENGTH; i += numVals)
    {

        pg = svwhilelt_b32_s32(i / 2, LENGTH / 2);
        pg_load = svzip1_b32(pg, pg); // svwhilelt_b32_s32(i, LENGTH);

        a = svld1_gather_index(pg_load, t + i, ind_load);
        b = svld1_gather_index(pg_load, t + i + BLOCK_SIZE / 2, ind_load);

        // printf("load\n");
        // svprint_ui(pg, ind_load, 16);
        // svprintf_f32(pg, a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, b, numVals, BLOCK_SIZE);

        // stage1
        // printf("stage 1\n");
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // svprintf_f32(pg, new_a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, new_b, numVals, BLOCK_SIZE);

        cmul_twiddle(&pg, &new_b, &twiddle_1, &new_bt);

        // svprintf_f32(pg, new_bt, numVals, BLOCK_SIZE);

        suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

        // svprintf_f32(pg, a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, b, numVals, BLOCK_SIZE);

        // stage2
        // printf("stage 2\n");
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // svprintf_f32(pg, new_a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, new_b, numVals, BLOCK_SIZE);

        cmul_twiddle(&pg, &new_b, &twiddle_2, &new_bt);

        // svprintf_f32(pg, new_bt, numVals, BLOCK_SIZE);

        suffle(&pg, &new_a, &new_bt, &ind_even, &ind_odd, &ind_zip, &a, &b);

        // svprintf_f32(pg, a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, b, numVals, BLOCK_SIZE);

        // stage3
        // printf("stage 3\n");
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // svprintf_f32(pg, new_a, numVals, BLOCK_SIZE);
        // svprintf_f32(pg, new_b, numVals, BLOCK_SIZE);

        // store
        //svst1_scatter_index(pg_load, f + i, ind_store, new_a);
        //svst1_scatter_index(pg_load, f + i + BLOCK_SIZE, ind_store, new_b);

        svst1_scatter_offset(pg_load, f + i/2*f_stride/4 + 0, offsets, new_a);
        svst1_scatter_offset(pg_load, f + i/2*f_stride/4 + f_stride, offsets, new_b); //todo fel
    }
   // printf("end\n");
}

//                      _1
static inline void fft8(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    float twiddle_factors[16] = {COS_0PI_OVER_4, SIN_0PI_OVER_4, COS_1PI_OVER_4, SIN_1PI_OVER_4, COS_2PI_OVER_4, SIN_2PI_OVER_4, COS_3PI_OVER_4, SIN_3PI_OVER_4,
                                 COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2, COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2};

    svbool_t pg, pg_load;

    svfloat32_t b, a, new_b, new_a, twiddle, new_bt;
    svuint32_t ind_a, ind_b;

    const uint64_t numVals = svcntw();

    // todo svdup no read
    const uint32_t index_zip[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const uint32_t index_low[4] = {0, 1, 2, 3};
    const uint32_t index_high[4] = {4, 5, 6, 7};
    const uint32_t index_even[4] = {0, 1, 4, 5};
    const uint32_t index_odd[4] = {2, 3, 6, 7};

    new_bt = svdup_f32(0.0f);
    // numvals >= 8
    for (uint32_t i = 0; i < 8; i += numVals)
    {

        pg = svwhilelt_b32_s32(i, 8);
        pg_load = svwhilelt_b32_s32(i / 2, 4);

        // svld1_vnum
        a = svzip1(svld1(pg_load, t + i / 2), svld1(pg_load, t + i / 2 + 8));
        b = svzip1(svld1(pg_load, t + i / 2 + 4), svld1(pg_load, t + i / 2 + 8 + 4));

        //printf("load\n");
        //svprint_f(pg, a, 8);
        //svprint_f(pg, b, 8);

        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        //printf("stage 1\n");
        //svprint_f(pg, new_a, 8);
        //svprint_f(pg, new_b, 8);

        twiddle = svld1(pg, twiddle_factors + i);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);

        //svprint_f(pg, new_bt, 8);

        const svuint32_t ind_zip = svld1(pg, index_zip + i);

        ind_a = svld1(pg, index_low + i);
        ind_b = svld1(pg, index_high + i);
        a = svtbl(svzip1(svtbl(new_a, ind_a), svtbl(new_bt, ind_a)), ind_zip);
        b = svtbl(svzip1(svtbl(new_a, ind_b), svtbl(new_bt, ind_b)), ind_zip);
        new_bt = svdup_f32(0.0f);

        //svprint_f(pg, a, 8);
        //svprint_f(pg, b, 8);

        //printf("stage 2\n");
        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        //svprint_f(pg, new_a, 8);
        //svprint_f(pg, new_b, 8);

        twiddle = svld1(pg, twiddle_factors + i + 8);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 0);
        new_bt = svcmla_m(pg, new_bt, twiddle, new_b, 270);

        //svprint_f(pg, new_bt, 8);

        ind_a = svld1(pg, index_even + i);
        ind_b = svld1(pg, index_odd + i);
        a = svtbl(svzip1(svtbl(new_a, ind_a), svtbl(new_bt, ind_a)), ind_zip);
        b = svtbl(svzip1(svtbl(new_a, ind_b), svtbl(new_bt, ind_b)), ind_zip);

        //svprint_f(pg, a, 8);
        //svprint_f(pg, b, 8);

        //printf("stage 3\n");
        new_a = svadd_m(pg, a, b);
        new_b = svsub_m(pg, a, b);

        //svprint_f(pg, new_a, 8);
        //svprint_f(pg, new_b, 8);

        svst1(pg, f + i + 8, new_b);
        svst1(pg, f + i, new_a);
    }
    //printf("done\n");
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