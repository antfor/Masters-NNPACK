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



//                      _256
static inline void sve_fft8x8_complex(
    const float t[restrict static 16 * 4],
    float f[restrict static 16 * 4],
    size_t f_stride)
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width; 
  //  const uint64_t numVals = qa * 2; // a and b
    const int dim = 2; //complex number 
    const uint64_t numVals = svcntw() * dim; // a and b

    const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);
    // const svuint32_t ind_load = index8(0, 8, 1, 9, 2, 10, 3, 11, 16);
    const int to_bytes = 4; 
    const svuint32_t ind_load = index8(to_bytes * 0, to_bytes * 8, to_bytes * 1, to_bytes * 9, to_bytes * 2, to_bytes * 10, to_bytes * 3, to_bytes * 11, to_bytes * 16);
    
     //const svuint32_t offsets = index8(4 * f_stride * 0 + 0, 4 * f_stride * 0 + 4, 4 * f_stride * 1 + 0, 4 * f_stride * 1 + 4, 4 * f_stride * 2 + 0, 4 * f_stride * 2 + 4, 4 * f_stride * 3 + 0, 4 * f_stride * 3 + 4, 4 * f_stride * BLOCK_SIZE);
     const svuint32_t offsets = soa_offset(simd_width, f_stride, BLOCK_SIZE);

    for (uint32_t i = 0; i < LENGTH; i += numVals)
    {

        pg = svwhilelt_b32_s32(i / dim, LENGTH / dim);
        pg_load = svzip1_b32(pg, pg); // svwhilelt_b32_s32(i, LENGTH);

        a = svld1_gather_offset(pg_load, t + i, ind_load);
        b = svld1_gather_offset(pg_load, t + i + BLOCK_SIZE / 2, ind_load);

        // stage1
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmul_twiddle(&pg, &new_b, &twiddle_1, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);

        // stage2
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmul_twiddle(&pg, &new_b, &twiddle_2, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_even, &ind_odd, &ind_zip, &a, &b);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // store
        //svst1_scatter_offset(pg_load, f + i / 2 * f_stride + 0, offsets, new_a);
        //svst1_scatter_offset(pg_load, f + i / 2 * f_stride + f_stride * 4, offsets, new_b);
        svst1_scatter_offset(pg_load, f + i / simd_width / 2 * f_stride + 0, offsets, new_a);
        svst1_scatter_offset(pg_load, f + i / simd_width / 2 * f_stride + f_stride * BLOCK_SIZE/2/simd_width + BLOCK_SIZE % (simd_width * 2), offsets, new_b);
    }
}


static inline void sve_ifft8x8_complex(
    const float t[restrict static 16 * 4],
    float f[restrict static 16 * 4],
    size_t f_stride)
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;

    const svfloat32_t twiddle_1 = svzip1(svdupq_f32(COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4), svdupq_f32(SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4));
    const svfloat32_t twiddle_2 = svdupq_f32(COS_0PI_OVER_2, SIN_0PI_OVER_2, COS_1PI_OVER_2, SIN_1PI_OVER_2);

    svbool_t pg, pg_load;
    svfloat32_t b, a, new_b, new_a, new_bt;

    const int simd_width = nnp_hwinfo.simd_width; 
  //  const uint64_t numVals = qa * 2; // a and b
    const int dim = 2; //complex number 
    const uint64_t numVals = svcntw() * dim; // a and b

    const svuint32_t ind_zip = index8(0, 2, 4, 6, 1, 3, 5, 7, 8);
    const svuint32_t ind_low = index4(0, 1, 2, 3, 8);
    const svuint32_t ind_high = index4(4, 5, 6, 7, 8);
    const svuint32_t ind_even = index4(0, 1, 4, 5, 8);
    const svuint32_t ind_odd = index4(2, 3, 6, 7, 8);
    
    const svuint32_t offsets = soa_offset(simd_width, f_stride, BLOCK_SIZE);

    for (uint32_t i = 0; i < LENGTH; i += numVals)
    {

        pg = svwhilelt_b32_s32(i / dim, LENGTH / dim);
        pg_load = svzip1_b32(pg, pg); // svwhilelt_b32_s32(i, LENGTH);

        a = svld1_gather_offset(pg_load, t + i, offsets);
        b = svld1_gather_offset(pg_load, t + i + BLOCK_SIZE / 2, offsets);

        // stage3
        butterfly(&pg, &a, &b, &new_a, &new_b);

        // stage2
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmul_twiddle(&pg, &new_b, &twiddle_2, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_even, &ind_odd, &ind_zip, &a, &b);

        // stage1
        butterfly(&pg, &a, &b, &new_a, &new_b);
        cmul_twiddle(&pg, &new_b, &twiddle_1, &new_bt);
        suffle(&pg, &new_a, &new_bt, &ind_low, &ind_high, &ind_zip, &a, &b);


        // store
        //svst1_scatter_offset(pg_load, f + i / 2 * f_stride + 0, offsets, new_a);
        //svst1_scatter_offset(pg_load, f + i / 2 * f_stride + f_stride * 4, offsets, new_b);
        svst1(pg_load, f + i, new_a);
        svst1(pg_load, f + i + LENGTH / 2, new_b); 
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