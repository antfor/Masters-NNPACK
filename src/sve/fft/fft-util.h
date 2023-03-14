#pragma once

#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>



//--fft--------------------------------------------------------------

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



