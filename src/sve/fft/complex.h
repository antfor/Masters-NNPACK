#pragma once

#include <nnpack/fft-constants.h>
#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>


static inline void svprint_f(svbool_t pg, svfloat32_t printme, const int n){

	float32_t tmp[n];

	svst1(pg, tmp, printme);

	for(int i =0; i< n; i++)
		printf("%f ", tmp[i]);

	printf("\n");
}

static inline void svprint_i(svbool_t pg, svint32_t printme, const int n){

	int32_t tmp[n];

	svst1(pg, tmp, printme);

	for(int i =0; i< n; i++)
		printf("%d ", tmp[i]);

	printf("\n");
}

inline static void fft4(
    const float t[restrict static 8],
    float f[restrict static 8])
{
    
}



static inline void fft8(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    //const float f8t0[16] = {1.0f,0.0f,  1.0f,  0.0f,                 1.0f,  0.0f,  1.0f,  0.0f,                  1.0f,0.0f,  1.0f,  0.0f,                   1.0f, 0.0f, 1.0f, 0.0f};
    const svfloat32_t f0t = svdupq_f32(1.0f,0.0f,1.0f,0.0f);
    const float f8t1[16] = {1.0f,0.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f,0.0f, -SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, 1.0f,   SQRT2_OVER_2,SQRT2_OVER_2};
    const float f8t2[16] = {1.0f,0.0f,  0.0f, -1.0f,                -1.0f,  0.0f,  0.0f,  1.0f,                  1.0f,0.0f,  0.0f, -1.0f,                  -1.0f, 0.0f, 0.0f, 1.0f};
    const float f8t3[16] = {1.0f,0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f,  1.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f,0.0f,  SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, -1.0f, -SQRT2_OVER_2, SQRT2_OVER_2};
    const float f8t4[16] = {1.0f,0.0f, -1.0f,  0.0f,                 1.0f,  0.0f, -1.0f,  0.0f,                  1.0f,0.0f, -1.0f,  0.0f,                   1.0f, 0.0f,-1.0f, 0.0f};
    const float f8t5[16] = {1.0f,0.0f, -SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, -1.0f,  SQRT2_OVER_2,  SQRT2_OVER_2, -1.0f,0.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, 1.0f,  -SQRT2_OVER_2, -SQRT2_OVER_2};
    const float f8t6[16] = {1.0f,0.0f,  0.0f,  1.0f,                -1.0f,  0.0f,  0.0f, -1.0f,                  1.0f,0.0f,  0.0f,  1.0f,                  -1.0f, 0.0f, 0.0f,-1.0f};
    const float f8t7[16] = {1.0f,0.0f,  SQRT2_OVER_2, SQRT2_OVER_2,  0.0f,  1.0f, -SQRT2_OVER_2,  SQRT2_OVER_2, -1.0f,0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f,  SQRT2_OVER_2, -SQRT2_OVER_2};

    svfloat32_t f0, f1 ,f2, f3, f4, f5, f6, f7;
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
    const svbool_t r_active = svdupq_b32(1,0,1,0);
	const svbool_t i_active = svdupq_b32(0,1,0,1);

    uint64_t numVals = svlen(f0);
    const svint32_t ind = svzip1(svindex_s32(0,1*4), svindex_s32(8*4,1*4)); 

    for (uint32_t i = 0; i < 16; i += numVals)
    {
        pg = svwhilelt_b32_s32(i, 16);

        const svfloat32_t fd = svld1_gather_offset(pg, &t[i/2],ind);
    
        
        f0 = svcmla_m(pg, f0, f0t, fd, 0);
        f0 = svcmla_m(pg, f0, f0t, fd, 90);

        const svfloat32_t f1t = svld1(pg, &f8t1[i]);
        f1 = svcmla_m(pg, f1, f1t, fd, 0);
        f1 = svcmla_m(pg, f1, f1t, fd, 90);

        const svfloat32_t f2t = svld1(pg, &f8t2[i]);
        f2 = svcmla_m(pg, f2, f2t, fd, 0);
        f2 = svcmla_m(pg, f2, f2t, fd, 90);

        const svfloat32_t f3t = svld1(pg, &f8t3[i]);
        f3 = svcmla_m(pg, f3, f3t, fd, 0);
        f3 = svcmla_m(pg, f3, f3t, fd, 90);

        const svfloat32_t f4t = svld1(pg, &f8t4[i]);
        f4 = svcmla_m(pg, f4, f4t, fd, 0);
        f4 = svcmla_m(pg, f4, f4t, fd, 90);

        const svfloat32_t f5t = svld1(pg, &f8t5[i]);
        f5 = svcmla_m(pg, f5, f5t, fd, 0);
        f5 = svcmla_m(pg, f5, f5t, fd, 90);

        const svfloat32_t f6t = svld1(pg, &f8t6[i]);
        f6 = svcmla_m(pg, f6, f6t, fd, 0);
        f6 = svcmla_m(pg, f6, f6t, fd, 90);

        const svfloat32_t f7t = svld1(pg, &f8t7[i]);
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