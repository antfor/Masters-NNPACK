#pragma once

#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#include<stdlib.h>


//--fft--------------------------------------------------------------

static inline void butterfly(svbool_t *pg, svfloat32_t *a, svfloat32_t *b, svfloat32_t *new_a, svfloat32_t *new_b)
{
    *new_a = svadd_m(*pg, *a, *b);
    *new_b = svsub_m(*pg, *a, *b);
}

static inline void mulc_twiddle(svbool_t *pg, svfloat32_t *br, svfloat32_t *bi, const svfloat32_t *tr, const svfloat32_t *ti, svfloat32_t *new_br, svfloat32_t *new_bi)
{
    *new_br = svadd_m(*pg, svmul_m(*pg, *tr, *br), svmul_m(*pg, *ti, *bi));
    *new_bi = svsub_m(*pg, svmul_m(*pg, *tr, *bi), svmul_m(*pg, *ti, *br));
}

static inline void cmulc_twiddle(svbool_t *pg, svfloat32_t *b, const svfloat32_t *t, svfloat32_t *new_b)
{
    *new_b = svdup_f32(0.0f);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 0);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 270);
}

static inline void mul_twiddle(svbool_t *pg, svfloat32_t *br, svfloat32_t *bi, const svfloat32_t *tr, const svfloat32_t *ti, svfloat32_t *new_br, svfloat32_t *new_bi)
{
    *new_br = svsub_m(*pg, svmul_m(*pg, *tr, *br), svmul_m(*pg, *ti, *bi));
    *new_bi = svadd_m(*pg, svmul_m(*pg, *tr, *bi), svmul_m(*pg, *ti, *br));
}

static inline void cmul_twiddle(svbool_t *pg, svfloat32_t *b, const svfloat32_t *t, svfloat32_t *new_b)
{
    *new_b = svdup_f32(0.0f);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 0);
    *new_b = svcmla_m(*pg, *new_b, *t, *b, 90);
}

static inline void suffle(svbool_t *pg, svfloat32_t *a, svfloat32_t *b, const svuint32_t *ind_a, const svuint32_t *ind_b, const svuint32_t *ind_zip, svfloat32_t *new_a, svfloat32_t *new_b)
{

    *new_a = svtbl(svzip1(svtbl(*a, *ind_a), svtbl(*b, *ind_a)), *ind_zip);
    *new_b = svtbl(svzip1(svtbl(*a, *ind_b), svtbl(*b, *ind_b)), *ind_zip);
}

//--ifft---------------------------------------------------------------

static inline void zip_rows_8(float block[restrict static 1]){

	const uint32_t BLOCK_SIZE = 8;
	const uint64_t numVals = svcntw();
	svbool_t pg;
	svfloat32_t r1, r2, r3, r4, r5, r6;

	for(int i = 0; i < BLOCK_SIZE; i+= numVals){

		pg = svwhilelt_b32_s32(i, BLOCK_SIZE);

		r1 = svld1(pg, block + i + BLOCK_SIZE * 1);
		r2 = svld1(pg, block + i + BLOCK_SIZE * 2);
		r3 = svld1(pg, block + i + BLOCK_SIZE * 3);

		r4 = svld1(pg, block + i + BLOCK_SIZE * 4);
		r5 = svld1(pg, block + i + BLOCK_SIZE * 5);
		r6 = svld1(pg, block + i + BLOCK_SIZE * 6);


		svst1(pg, block + i + BLOCK_SIZE * 1, r4);
		svst1(pg, block + i + BLOCK_SIZE * 2, r1);
		svst1(pg, block + i + BLOCK_SIZE * 3, r5);

		svst1(pg, block + i + BLOCK_SIZE * 4, r2);
		svst1(pg, block + i + BLOCK_SIZE * 5, r6);
		svst1(pg, block + i + BLOCK_SIZE * 6, r3);
		
	}
}

//--index--------------------------------------------------------------

static inline svuint32_t index2(uint32_t a, uint32_t b, uint32_t step)
{
    return svzip1(svindex_u32(a, step), svindex_u32(b, step));
}

static inline svuint32_t index4(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t step)
{
    return svzip1(index2(a, c, step), index2(b, d, step));
}

static inline svuint32_t index8(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7, uint32_t step)
{
    return svzip1(index4(i0, i2, i4, i6, step), index4(i1, i3, i5, i7, step));
}


static inline svuint32_t indexN(uint32_t *ab, uint32_t N, uint32_t step){

    switch (N)
    {
    case 2:
        return index2(ab[0], ab[1], step);
    case 4:
        return index4(ab[0], ab[1], ab[2], ab[3], step);
    case 8:
        return index8(ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6], ab[7], step);
    default:
        printf("indexN todo add index for N=%d\n", N);
        exit(0.0);
        break;
    }
}


//----gp-utils-----------------------------------------------------------------

static inline int imin(int a, int b){
	return a < b ? a : b;
}


static inline int imax(int a, int b){
	return a > b ? a : b;
}

