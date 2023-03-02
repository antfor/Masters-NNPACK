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


//const float f8t[16] = {1.0f,0.0f, SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, 1.0f, SQRT2_OVER_2, SQRT2_OVER_2, 1.0f,0.0f, SQRT2_OVER_2, SQRT2_OVER_2, 0.0f, 1.0f, SQRT2_OVER_2,SQRT2_OVER_2};
//const float f8t6[16] = {1.0f,0.0f,  0.0f,  1.0f,                1.0f,  0.0f,  0.0f, 1.0f,                  1.0f,0.0f,  0.0f,  1.0f,                  1.0f, 0.0f, 0.0f,1.0f};
//const float f8t4[16] = {1.0f,0.0f,  1.0f,  0.0f,                 1.0f,  0.0f, 1.0f,  0.0f,                  1.0f,0.0f, 1.0f,  0.0f,                   1.0f, 0.0f,1.0f, 0.0f};



static inline void fft8(
    const float t[restrict static 16],
    float f[restrict static 16])
{

    const float f8t1[16] = {1.0f,0.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f,0.0f, -SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, 1.0f,   SQRT2_OVER_2,SQRT2_OVER_2};
    const float f8t3[16] = {1.0f,0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f,  1.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, -1.0f,0.0f,  SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, -1.0f, -SQRT2_OVER_2, SQRT2_OVER_2};
    const float f8t5[16] = {1.0f,0.0f, -SQRT2_OVER_2, SQRT2_OVER_2,  0.0f, -1.0f,  SQRT2_OVER_2,  SQRT2_OVER_2, -1.0f,0.0f,  SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, 1.0f,  -SQRT2_OVER_2, -SQRT2_OVER_2};
    const float f8t7[16] = {1.0f,0.0f,  SQRT2_OVER_2, SQRT2_OVER_2,  0.0f,  1.0f, -SQRT2_OVER_2,  SQRT2_OVER_2, -1.0f,0.0f, -SQRT2_OVER_2, -SQRT2_OVER_2, 0.0f, -1.0f,  SQRT2_OVER_2, -SQRT2_OVER_2};

    const float f8t2[16] = {1.0f,0.0f,  0.0f, -1.0f,                -1.0f,  0.0f,  0.0f,  1.0f,                  1.0f,0.0f,  0.0f, -1.0f,                  -1.0f, 0.0f, 0.0f, 1.0f};
    const float f8t4[16] = {1.0f,0.0f, -1.0f,  0.0f,                 1.0f,  0.0f, -1.0f,  0.0f,                  1.0f,0.0f, -1.0f,  0.0f,                   1.0f, 0.0f,-1.0f, 0.0f};
    const float f8t6[16] = {1.0f,0.0f,  0.0f,  1.0f,                -1.0f,  0.0f,  0.0f, -1.0f,                  1.0f,0.0f,  0.0f,  1.0f,                  -1.0f, 0.0f, 0.0f,-1.0f};

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
      

        f0 = svld1_gather_offset(pg, t + i,ind);

        const svfloat32_t f1t = svld1(pg, f8t1 + i);
        f1 = svcmla_m(pg, f1, f1t, f0, 0);
        f1 = svcmla_m(pg, f1, f1t, f0, 90);

        const svfloat32_t f2t = svld1(pg, f8t2 + i);
        f2 = svcmla_m(pg, f2, f2t, f0, 0);
        f2 = svcmla_m(pg, f2, f2t, f0, 90);

        const svfloat32_t f3t = svld1(pg, f8t3 + i);
        f3 = svcmla_m(pg, f3, f3t, f0, 0);
        f3 = svcmla_m(pg, f3, f3t, f0, 90);

        const svfloat32_t f4t = svld1(pg, f8t4 + i);
        f4 = svcmla_m(pg, f4, f4t, f0, 0);
        f4 = svcmla_m(pg, f4, f4t, f0, 90);

        const svfloat32_t f5t = svld1(pg, f8t5 + i);
        f5 = svcmla_m(pg, f5, f5t, f0, 0);
        f5 = svcmla_m(pg, f5, f5t, f0, 90);

        const svfloat32_t f6t = svld1(pg, f8t6 + i);
        f6 = svcmla_m(pg, f6, f6t, f0, 0);
        f6 = svcmla_m(pg, f6, f6t, f0, 90);

        const svfloat32_t f7t = svld1(pg, f8t7 + i);
        f7 = svcmla_m(pg, f7, f7t, f0, 0);
        f7 = svcmla_m(pg, f7, f7t, f0, 90);

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

static inline void fft8_new(
    const float t[restrict static 16],
    float f[restrict static 16])
{
    
    const float f8t1r[8] = {1.0f,  SQRT2_OVER_2, 0.0f, -SQRT2_OVER_2, -1.0f, -SQRT2_OVER_2, 0.0f,  SQRT2_OVER_2};
    const float f8t3r[8] = {1.0f, -SQRT2_OVER_2, 0.0f,  SQRT2_OVER_2, -1.0f,  SQRT2_OVER_2, 0.0f, -SQRT2_OVER_2};
    const float f8t5r[8] = {1.0f, -SQRT2_OVER_2, 0.0f,  SQRT2_OVER_2, -1.0f,  SQRT2_OVER_2, 0.0f, -SQRT2_OVER_2};
    const float f8t7r[8] = {1.0f,  SQRT2_OVER_2, 0.0f, -SQRT2_OVER_2, -1.0f, -SQRT2_OVER_2, 0.0f,  SQRT2_OVER_2};

    const float f8t2r[8] = {1.0f,  0.0f,                 -1.0f,  0.0f,                  1.0f,  0.0f,                  -1.0f, 0.0f};
    const float f8t4r[8] = {1.0f, -1.0f,                  1.0f, -1.0f,                  1.0f, -1.0f,                   1.0f,-1.0f};
    const float f8t6r[8] = {1.0f,  0.0f,                 -1.0f,  0.0f,                  1.0f,  0.0f,                  -1.0f, 0.0f};

    const float f8t1i[8] = {0.0f, -SQRT2_OVER_2, -1.0f,  -SQRT2_OVER_2,0.0f, SQRT2_OVER_2, 1.0f,  SQRT2_OVER_2};
    const float f8t3i[8] = {0.0f, -SQRT2_OVER_2,  1.0f,  -SQRT2_OVER_2,0.0f, SQRT2_OVER_2, -1.0f, SQRT2_OVER_2};
    const float f8t5i[8] = {0.0f,  SQRT2_OVER_2, -1.0f,   SQRT2_OVER_2,0.0f,-SQRT2_OVER_2, 1.0f,  -SQRT2_OVER_2};
    const float f8t7i[8] = {0.0f,  SQRT2_OVER_2,  1.0f,   SQRT2_OVER_2,0.0f,-SQRT2_OVER_2, -1.0f, -SQRT2_OVER_2};

    const float f8t2i[8] = {0.0f, -1.0f,                0.0f,  1.0f,                  0.0f, -1.0f,                   0.0f, 1.0f};
    const float f8t4i[8] = {0.0f,  0.0f,                0.0f,  0.0f,                  0.0f,  0.0f,                   0.0f, 0.0f};
    const float f8t6i[8] = {0.0f,  1.0f,                0.0f, -1.0f,                  0.0f,  1.0f,                   0.0f,-1.0f};


    svfloat32_t f0r, f1r ,f2r, f3r, f4r, f5r, f6r, f7r,f0i, f1i ,f2i, f3i, f4i, f5i, f6i, f7i;
    f0r = svdup_f32(0.0f);
    f1r = svdup_f32(0.0f);
    f2r = svdup_f32(0.0f);
    f3r = svdup_f32(0.0f);
    f4r = svdup_f32(0.0f);
    f5r = svdup_f32(0.0f);
    f6r = svdup_f32(0.0f);
    f7r = svdup_f32(0.0f);

    f0i = svdup_f32(0.0f);
    f1i = svdup_f32(0.0f);
    f2i = svdup_f32(0.0f);
    f3i = svdup_f32(0.0f);
    f4i = svdup_f32(0.0f);
    f5i = svdup_f32(0.0f);
    f6i = svdup_f32(0.0f);
    f7i = svdup_f32(0.0f);

    svfloat32_t f1tr, f1ti, f2tr, f2ti, f3tr, f3ti, f4tr, f4ti, f5tr, f5ti, f6tr, f6ti, f7tr, f7ti;
    svbool_t pg;
    const svbool_t all_active = svptrue_b32();

    uint64_t numVals = svlen(f0r);

    for (uint32_t i = 0; i < 8; i += numVals)
    {
        pg = svwhilelt_b32_s32(i, 8);
      

        f0r = svld1(pg, t + i);
        f0i = svld1(pg, t + i + 8);

        f1tr = svld1(pg, f8t1r + i);
        f1ti = svld1(pg, f8t1i + i);
        f1r = svmad_m(pg, f0r, f1tr, f1r);
        f1r = svmsb_m(pg, f0i, f1ti, f1r);
        f1i = svmad_m(pg, f0r, f1ti, f1i);
        f1i = svmad_m(pg, f0i, f1tr, f1i);
        

        f2tr = svld1(pg, f8t2r + i);
        f2ti = svld1(pg, f8t2i + i);
        f2r = svmad_m(pg, f0r, f2tr, f2r);
        f2r = svmsb_m(pg, f0i, f2ti, f2r);
        f2i = svmad_m(pg, f0r, f2ti, f2i);
        f2i = svmad_m(pg, f0i, f2tr, f2i);


        f3tr = svld1(pg, f8t3r + i);
        f3ti = svld1(pg, f8t3i + i);
        f3r = svmad_m(pg, f0r, f3tr, f3r);
        f3r = svmsb_m(pg, f0i, f3ti, f3r);
        f3i = svmad_m(pg, f0r, f3ti, f3i);
        f3i = svmad_m(pg, f0i, f3tr, f3i);

        f4tr = svld1(pg, f8t4r + i);
        f4ti = svld1(pg, f8t4i + i);
        f4r = svmad_m(pg, f0r, f4tr, f4r);
        f4r = svmsb_m(pg, f0i, f4ti, f4r);
        f4i = svmad_m(pg, f0r, f4ti, f4i);
        f4i = svmad_m(pg, f0i, f4tr, f4i);


        f5tr = svld1(pg, f8t5r + i);
        f5ti = svld1(pg, f8t5i + i);
        f5r = svmad_m(pg, f0r, f5tr, f5r);
        f5r = svmsb_m(pg, f0i, f5ti, f5r);
        f5i = svmad_m(pg, f0r, f5ti, f5i);
        f5i = svmad_m(pg, f0i, f5tr, f5i);

        f6tr = svld1(pg, f8t6r + i);
        f6ti = svld1(pg, f8t6i + i);
        f6r = svmad_m(pg, f0r, f6tr, f6r);
        f6r = svmsb_m(pg, f0i, f6ti, f6r);
        f6i = svmad_m(pg, f0r, f6ti, f6i);
        f6i = svmad_m(pg, f0i, f6tr, f6i);


        f7tr = svld1(pg, f8t7r + i);
        f7ti = svld1(pg, f8t7i + i);
        f7r = svmad_m(pg, f0r, f7tr, f7r);
        f7r = svmsb_m(pg, f0i, f7ti, f7r);
        f7i = svmad_m(pg, f0r, f7ti, f7i);
        f7i = svmad_m(pg, f0i, f7tr, f7i);

    }

    f[0] = svaddv(all_active, f0r);
    f[1] = svaddv(all_active, f0i);

    f[2] = svaddv(all_active, f1r);
    f[3] = svaddv(all_active, f1i);

    f[4] = svaddv(all_active, f2r);
    f[5] = svaddv(all_active, f2i);

    f[6] = svaddv(all_active, f3r);
    f[7] = svaddv(all_active, f3i);

    f[8] = svaddv(all_active, f4r);
    f[9] = svaddv(all_active, f4i);

    f[10] = svaddv(all_active, f5r);
    f[11] = svaddv(all_active, f5i);

    f[12] = svaddv(all_active, f6r);
    f[13] = svaddv(all_active, f6i);

    f[14] = svaddv(all_active, f7r);
    f[15] = svaddv(all_active, f7i);

}


inline static void fft16(
    const float t[restrict static 32],
    float f[restrict static 32])
{
}