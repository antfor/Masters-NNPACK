#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>


void svprint_b(svbool_t pg, svbool_t printme){



}

void svprint_f(svbool_t pg, svfloat32_t printme, const int n){

	float32_t tmp[n];

	svst1(pg, tmp, printme);

	for(int i =0; i< n; i++)
		printf("%f ", tmp[i]);

	printf("\n");
}


void nnp_cgemm_conjb_only_2x2__scalar_new_old(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t svAcc00r, svAcc00i,svAcc01r, svAcc01i,svAcc10r, svAcc10i,svAcc11r, svAcc11i;
	svfloat32_t a0r,a0i,a1r,a1i,b0r,b0i,b1r,b1i;
	svbool_t p0,p1,p2,p3;
	svAcc00r = svdup_f32(0.0f);
	svAcc01r = svdup_f32(0.0f);
	svAcc10r = svdup_f32(0.0f);
	svAcc11r = svdup_f32(0.0f);

	svAcc00i = svdup_f32(0.0f);
	svAcc01i = svdup_f32(0.0f);
	svAcc10i = svdup_f32(0.0f);
	svAcc11i = svdup_f32(0.0f);
	const svbool_t all_active = svptrue_b32();
	const svint32_t ind = svindex_s32(0,4*4);
	uint64_t numVals = svlen(svAcc00r);

	for(uint32_t i = 0; i < k; i +=numVals){
	
		p0 = svwhilelt_b32_s32(i*4 + 0, k*4);


		a0r = svld1_gather_offset(p0, &A[4*i+0], ind);
		a0i = svld1_gather_offset(p0, &A[4*i+1], ind);

		a1r = svld1_gather_offset(p0, &A[4*i+2], ind);
		a1i = svld1_gather_offset(p0, &A[4*i+3], ind);

		b0r = svld1_gather_offset(p0, &B[4*i+0], ind);
		b0i = svld1_gather_offset(p0, &B[4*i+1], ind);

		b1r = svld1_gather_offset(p0, &B[4*i+2], ind);
		b1i = svld1_gather_offset(p0, &B[4*i+3], ind);

/*
		int si = 8;
		svprint_f(p0,a0r,si);
		svprint_f(p0,a0i,si);
		svprint_f(p0,a1r,si);
		svprint_f(p0,a1i,si);

		svprint_f(p0,b0r,si);
		svprint_f(p0,b0i,si);
		svprint_f(p0,b1r,si);
		svprint_f(p0,b1i,si);
*/
		svAcc00r = svmad_m(p0, a0r, b0r, svAcc00r);
		svAcc00r = svmad_m(p0, a0i, b0i, svAcc00r); 
		svAcc00i = svmad_m(p0, a0i, b0r, svAcc00i);
		svAcc00i = svmsb_m(p0, a0r, b0i, svAcc00i); 

		svAcc01r = svmad_m(p0, a0r, b1r, svAcc01r);
		svAcc01r = svmad_m(p0, a0i, b1i, svAcc01r);
		svAcc01i = svmad_m(p0, a0i, b1r, svAcc01i);
		svAcc01i = svmsb_m(p0, a0r, b1i, svAcc01i);

		svAcc10r = svmad_m(p0, a1r, b0r, svAcc10r);
		svAcc10r = svmad_m(p0, a1i, b0i, svAcc10r); 
		svAcc10i = svmad_m(p0, a1i, b0r, svAcc10i);
		svAcc10i = svmsb_m(p0, a1r, b0i, svAcc10i); 

		svAcc11r = svmad_m(p0, a1r, b1r, svAcc11r);
		svAcc11r = svmad_m(p0, a1i, b1i, svAcc11r); 
		svAcc11i = svmad_m(p0, a1i, b1r, svAcc11i);
		svAcc11i = svmsb_m(p0, a1r, b1i, svAcc11i); 

	}

	float32_t acc00r = svaddv_f32(all_active, svAcc00r);
	float32_t acc00i = svaddv_f32(all_active, svAcc00i);

	float32_t acc01r = svaddv_f32(all_active, svAcc01r);
	float32_t acc01i = svaddv_f32(all_active, svAcc01i);

	float32_t acc10r = svaddv_f32(all_active, svAcc10r);
	float32_t acc10i = svaddv_f32(all_active, svAcc10i);

	float32_t acc11r = svaddv_f32(all_active, svAcc11r);
	float32_t acc11i = svaddv_f32(all_active, svAcc11i);


	if (update != 0) {
		C[0] += acc00r;
		C[1] += acc00i;
		C[2] += acc01r;
		C[3] += acc01i;
		C += row_stride_c;
		C[0] += acc10r;
		C[1] += acc10i;
		C[2] += acc11r;
		C[3] += acc11i;
	} else {
		C[0] = acc00r;
		C[1] = acc00i;
		C[2] = acc01r;
		C[3] = acc01i;
		C += row_stride_c;
		C[0] = acc10r;
		C[1] = acc10i;
		C[2] = acc11r;
		C[3] = acc11i;
	}

}


void nnp_cgemm_conjb_only_2x2__scalar(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t svAcc00, svAcc01,svAcc10,svAcc11;
	svfloat32_t a0,a1,b0,b1;
	svbool_t p0;

	const svbool_t r_active = svdupq_b32(1,0,1,0);
	const svbool_t i_active = svdupq_b32(1,0,1,0);
	const svint32_t ind =svzip1(svindex_s32(0,4*4),svindex_s32(1,4*4));

	uint64_t numVals = svlen(svAcc00);

	for(uint32_t i = 0; i < k; i +=numVals){
	
		p0 = svwhilelt_b32_s32(i*4 + 0, k*4);


		a0 = svld1_gather_offset(p0, &A[4*i+0], ind);
		a1 = svld1_gather_offset(p0, &A[4*i+1], ind);
		
		b0 = svld1_gather_offset(p0, &B[4*i+0], ind);
		b1 = svld1_gather_offset(p0, &B[4*i+1], ind);


		svAcc00 = svcmla_m(p0,svAcc00, b0, a0, 0);
		svAcc00 = svcmla_m(p0,svAcc00, b0, a0, 270); 

		svAcc01 = svcmla_m(p0,svAcc01, b1, a0, 0);
		svAcc01 = svcmla_m(p0,svAcc01, b1, a0, 270); 

		svAcc10 = svcmla_m(p0,svAcc10, b0, a1, 0);
		svAcc10 = svcmla_m(p0,svAcc10, b0, a1, 270); 

		svAcc11 = svcmla_m(p0,svAcc11, b1, a1, 0);
		svAcc11 = svcmla_m(p0,svAcc11, b1, a1, 270); 

	}

	float32_t acc00r = svaddv_f32(r_active, svAcc00);
	float32_t acc00i = svaddv_f32(i_active, svAcc00);

	float32_t acc01r = svaddv_f32(r_active, svAcc01);
	float32_t acc01i = svaddv_f32(i_active, svAcc01);

	float32_t acc10r = svaddv_f32(r_active, svAcc10);
	float32_t acc10i = svaddv_f32(i_active, svAcc10);

	float32_t acc11r = svaddv_f32(r_active, svAcc11);
	float32_t acc11i = svaddv_f32(i_active, svAcc11);


	if (update != 0) {
		C[0] += acc00r;
		C[1] += acc00i;
		C[2] += acc01r;
		C[3] += acc01i;
		C += row_stride_c;
		C[0] += acc10r;
		C[1] += acc10i;
		C[2] += acc11r;
		C[3] += acc11i;
	} else {
		C[0] = acc00r;
		C[1] = acc00i;
		C[2] = acc01r;
		C[3] = acc01i;
		C += row_stride_c;
		C[0] = acc10r;
		C[1] = acc10i;
		C[2] = acc11r;
		C[3] = acc11i;
	}

}

//*/

void nnp_cgemm_conjb_only_2x2__scalar_old(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float acc00r, acc01r, acc10r, acc11r;
	float acc00i, acc01i, acc10i, acc11i;
	acc00r = acc01r = acc10r = acc11r = 0.0f;
	acc00i = acc01i = acc10i = acc11i = 0.0f;
	do
	{
		const float a0r = a[0];
		const float a1r = a[2];
		const float a0i = a[1];
		const float a1i = a[3];
		a += 4;

		const float b0r = b[0];
		const float b0i = b[1];
		const float b1r = b[2];
		const float b1i = b[3];
		b += 4;

		acc00r += a0r * b0r;
		acc00r += a0i * b0i;
		acc00i += a0i * b0r;
		acc00i -= a0r * b0i;

		acc01r += a0r * b1r;
		acc01r += a0i * b1i;
		acc01i += a0i * b1r;
		acc01i -= a0r * b1i;

		acc10r += a1r * b0r;
		acc10r += a1i * b0i;
		acc10i += a1i * b0r;
		acc10i -= a1r * b0i;

		acc11r += a1r * b1r;
		acc11i += a1i * b1r;
		acc11r += a1i * b1i;
		acc11i -= a1r * b1i;

	} while (--k);

	if (update != 0)
	{
		c[0] += acc00r;
		c[1] += acc00i;
		c[2] += acc01r;
		c[3] += acc01i;
		c += row_stride_c;
		c[0] += acc10r;
		c[1] += acc10i;
		c[2] += acc11r;
		c[3] += acc11i;
	}
	else
	{
		c[0] = acc00r;
		c[1] = acc00i;
		c[2] = acc01r;
		c[3] = acc01i;
		c += row_stride_c;
		c[0] = acc10r;
		c[1] = acc10i;
		c[2] = acc11r;
		c[3] = acc11i;
	}
}

int main2()
{
	const int k = 2;
	float A[2*4] = {1, 2, 3, 4,1, 2, 3, 4};
	float B[2*4] = {1, 2, 3, 4,1, 2, 3, 4};
	float C[8] = {0, 0, 0, 0};

	nnp_cgemm_conjb_only_2x2__scalar_old(k, 0, A, B, C, 4);
	printf("new \n");
	printf("r: %f %f %f %f \n", C[0], C[1], C[2], C[3]);
	printf("i: %f %f %f %f \n", C[4], C[5], C[6], C[7]);
	nnp_cgemm_conjb_only_2x2__scalar(k, 0, A, B, C, 4);
	printf("old \n");
	printf("r: %f %f %f %f \n", C[0], C[1], C[2], C[3]);
	printf("i: %f %f %f %f \n", C[4], C[5], C[6], C[7]);
	// printf("Hello World\n");
	return 0;
}

void nnp_cgemm_conjb_upto_2x2__scalar(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float acc00r, acc01r, acc10r, acc11r;
	float acc00i, acc01i, acc10i, acc11i;
	acc00r = acc01r = acc10r = acc11r = 0.0f;
	acc00i = acc01i = acc10i = acc11i = 0.0f;
	do
	{
		const float a0r = a[0];
		const float a0i = a[1];
		a += 2;

		float a1r, a1i;
		if (mr > 1)
		{
			a1r = a[0];
			a1i = a[1];
			a += 2;
		}

		const float b0r = b[0];
		const float b0i = b[1];
		b += 2;

		acc00r += a0r * b0r;
		acc10r += a1r * b0r;
		acc00i += a0i * b0r;
		acc10i += a1i * b0r;

		acc00r += a0i * b0i;
		acc10r += a1i * b0i;
		acc00i -= a0r * b0i;
		acc10i -= a1r * b0i;

		if (nr > 1)
		{
			const float b1r = b[0];
			const float b1i = b[1];
			b += 2;

			acc01r += a0r * b1r;
			acc11r += a1r * b1r;
			acc01i += a0i * b1r;
			acc11i += a1i * b1r;

			acc01r += a0i * b1i;
			acc11r += a1i * b1i;
			acc01i -= a0r * b1i;
			acc11i -= a1r * b1i;
		}
	} while (--k);

	if (update != 0)
	{
		c[0] += acc00r;
		c[1] += acc00i;
		if (nr > 1)
		{
			c[2] += acc01r;
			c[3] += acc01i;
		}
		if (mr > 1)
		{
			c += row_stride_c;
			c[0] += acc10r;
			c[1] += acc10i;
			if (nr > 1)
			{
				c[2] += acc11r;
				c[3] += acc11i;
			}
		}
	}
	else
	{
		c[0] = acc00r;
		c[1] = acc00i;
		if (nr > 1)
		{
			c[2] = acc01r;
			c[3] = acc01i;
		}
		if (mr > 1)
		{
			c += row_stride_c;
			c[0] = acc10r;
			c[1] = acc10i;
			if (nr > 1)
			{
				c[2] = acc11r;
				c[3] = acc11i;
			}
		}
	}
}