#include <stddef.h>
#include <stdint.h>

#include <nnpack/arm_neon.h>
#include <sve/fft/sve-print.h>
#include <arm_sve.h>


void nnp_cVLgemm_conjb_only_2x2__sve(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t a0r, a1r, b0r, b1r;
	svfloat32_t a0i, a1i, b0i, b1i;

	svfloat32_t svAcc00r = svdup_f32(0.0f);
	svfloat32_t svAcc01r = svdup_f32(0.0f);
	svfloat32_t svAcc10r = svdup_f32(0.0f);
	svfloat32_t svAcc11r = svdup_f32(0.0f);

	svfloat32_t svAcc00i = svdup_f32(0.0f);
	svfloat32_t svAcc01i = svdup_f32(0.0f);
	svfloat32_t svAcc10i = svdup_f32(0.0f);
	svfloat32_t svAcc11i = svdup_f32(0.0f);


	const size_t simd_width = 8; //svcntw(); // min(svcntw()§
	const svbool_t p0 = svwhilelt_b32_s32(0, simd_width); // svptrue_b32();

	for(int i = 0; i < k; i++){

		a0r = svld1(p0, A + i * simd_width * 4 + simd_width * 0);
		a0i = svld1(p0, A + i * simd_width * 4 + simd_width * 1);

		a1r = svld1(p0, A + i * simd_width * 4 + simd_width * 2);
		a1i = svld1(p0, A + i * simd_width * 4 + simd_width * 3);


		b0r = svld1(p0, B + i * simd_width * 4 + simd_width * 0);
		b0i = svld1(p0, B + i * simd_width * 4 + simd_width * 1);
	
		b1r = svld1(p0, B + i * simd_width * 4 + simd_width * 2);
		b1i = svld1(p0, B + i * simd_width * 4 + simd_width * 3);

		svAcc00r  = svmla_m(p0, svAcc00r, a0r, b0r);
		svAcc00r  = svmla_m(p0, svAcc00r, a0i, b0i);
		svAcc00i  = svmla_m(p0, svAcc00i, a0i, b0r);
		svAcc00i  = svmsb_m(p0, a0r, b0i, svAcc00i);


		svAcc01r  = svmla_m(p0, svAcc01r, a0r, b1r);
		svAcc01r  = svmla_m(p0, svAcc01r, a0i, b1i);
		svAcc01i  = svmla_m(p0, svAcc01i, a0i, b1r);
		svAcc01i  = svmsb_m(p0, a0r, b1i, svAcc01i);


		svAcc10r  = svmla_m(p0, svAcc10r, a1r, b0r);
		svAcc10r  = svmla_m(p0, svAcc10r, a1i, b0i);
		svAcc10i  = svmla_m(p0, svAcc10i, a1i, b0r);
		svAcc10i  = svmsb_m(p0, a1r, b0i, svAcc10i);


		svAcc11r  = svmla_m(p0, svAcc11r, a1r, b1r);
		svAcc11r  = svmla_m(p0, svAcc11r, a1i, b1i);
		svAcc11i  = svmla_m(p0, svAcc11i, a1i, b1r);
		svAcc11i  = svmsb_m(p0, a1r, b1i, svAcc11i);

	}
	
	if(update != 0){

		svAcc00r = svadd_m(p0, svAcc00r, svld1(p0, C + simd_width * 0));
		svAcc00i = svadd_m(p0, svAcc00i, svld1(p0, C + simd_width * 1));
		svAcc01r = svadd_m(p0, svAcc01r, svld1(p0, C + simd_width * 2));
		svAcc01i = svadd_m(p0, svAcc01i, svld1(p0, C + simd_width * 3));
		
		svAcc10r = svadd_m(p0, svAcc10r, svld1(p0, C + row_stride_c + simd_width * 0));
		svAcc10i = svadd_m(p0, svAcc10i, svld1(p0, C + row_stride_c + simd_width * 1));
		svAcc11r = svadd_m(p0, svAcc11r, svld1(p0, C + row_stride_c + simd_width * 2));
		svAcc11i = svadd_m(p0, svAcc11i, svld1(p0, C + row_stride_c + simd_width * 3));
	}

	svst1(p0, C + simd_width * 0, svAcc00r);
	svst1(p0, C + simd_width * 1, svAcc00i);
	svst1(p0, C + simd_width * 2, svAcc01r);
	svst1(p0, C + simd_width * 3, svAcc01i);
	C += row_stride_c;
	svst1(p0, C + simd_width * 0, svAcc10r);
	svst1(p0, C + simd_width * 1, svAcc10i);
	svst1(p0, C + simd_width * 2, svAcc11r);
	svst1(p0, C + simd_width * 3, svAcc11i);
}




void nnp_c4gemm_conjb_only_2x2__neon_new(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t svAcc00, svAcc01, svAcc10, svAcc11;
	svfloat32_t a0, a1, b0, b1;
	svbool_t p0;
	svAcc00 = svdup_f32(0.0f);
	svAcc01 = svdup_f32(0.0f);
	svAcc10 = svdup_f32(0.0f);
	svAcc11 = svdup_f32(0.0f);

	const svbool_t r_active = svdupq_b32(1, 0, 1, 0);
	const svbool_t i_active = svdupq_b32(0, 1, 0, 1);

	const svint32_t ind = svzip1(svindex_s32(0, 4 * 4), svindex_s32(1 * 4, 4 * 4));

	uint64_t numVals = svlen(svAcc00);

	for (uint32_t i = 0; i < k; i += numVals / 2)
	{

		p0 = svwhilelt_b32_s32(i * 2, k * 2);

		a0 = svld1_gather_offset(p0, &A[4 * i + 0], ind);
		a1 = svld1_gather_offset(p0, &A[4 * i + 2], ind);

		b0 = svld1_gather_offset(p0, &B[4 * i + 0], ind);
		b1 = svld1_gather_offset(p0, &B[4 * i + 2], ind);

		svAcc00 = svcmla_m(p0, svAcc00, b0, a0, 0);
		svAcc00 = svcmla_m(p0, svAcc00, b0, a0, 270);

		svAcc01 = svcmla_m(p0, svAcc01, b1, a0, 0);
		svAcc01 = svcmla_m(p0, svAcc01, b1, a0, 270);

		svAcc10 = svcmla_m(p0, svAcc10, b0, a1, 0);
		svAcc10 = svcmla_m(p0, svAcc10, b0, a1, 270);

		svAcc11 = svcmla_m(p0, svAcc11, b1, a1, 0);
		svAcc11 = svcmla_m(p0, svAcc11, b1, a1, 270);
	}

	float32_t acc00r = svaddv_f32(r_active, svAcc00);
	float32_t acc00i = svaddv_f32(i_active, svAcc00);

	float32_t acc01r = svaddv_f32(r_active, svAcc01);
	float32_t acc01i = svaddv_f32(i_active, svAcc01);

	float32_t acc10r = svaddv_f32(r_active, svAcc10);
	float32_t acc10i = svaddv_f32(i_active, svAcc10);

	float32_t acc11r = svaddv_f32(r_active, svAcc11);
	float32_t acc11i = svaddv_f32(i_active, svAcc11);

	if (update != 0)
	{
		C[0] += acc00r;
		C[1] += acc00i;
		C[2] += acc01r;
		C[3] += acc01i;
		C += row_stride_c;
		C[0] += acc10r;
		C[1] += acc10i;
		C[2] += acc11r;
		C[3] += acc11i;
	}
	else
	{
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


void nnp_c4gemm_conjb_upto_2x2__neon_new(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t svAcc00, svAcc01, svAcc10, svAcc11;
	svfloat32_t a0, a1, b0, b1;
	svbool_t p0;
	svAcc00 = svdup_f32(0.0f);
	svAcc01 = svdup_f32(0.0f);
	svAcc10 = svdup_f32(0.0f);
	svAcc11 = svdup_f32(0.0f);

	int sizeA = 2 * mr;
	int sizeB = 2 * nr;

	const svbool_t r_active = svdupq_b32(1, 0, 1, 0);
	const svbool_t i_active = svdupq_b32(0, 1, 0, 1);

	const svint32_t indA = svzip1(svindex_s32(0, sizeA * 4), svindex_s32(1 * 4, sizeA * 4));
	const svint32_t indB = svzip1(svindex_s32(0, sizeB * 4), svindex_s32(1 * 4, sizeB * 4));

	uint64_t numVals = svlen(svAcc00);

	for (uint32_t i = 0; i < k; i += numVals / 2)
	{

		p0 = svwhilelt_b32_s32(i * 2, k * 2);

		a0 = svld1_gather_offset(p0, &A[sizeA * i + 0], indA);

		if (mr > 1)
		{
			a1 = svld1_gather_offset(p0, &A[sizeA * i + 2], indA);
		}

		b0 = svld1_gather_offset(p0, &B[sizeB * i + 0], indB);

		if (nr > 1)
		{
			b1 = svld1_gather_offset(p0, &B[sizeB * i + 2], indB);
		}

		svAcc00 = svcmla_m(p0, svAcc00, b0, a0, 0);
		svAcc00 = svcmla_m(p0, svAcc00, b0, a0, 270);

		if (mr > 1)
		{
			svAcc10 = svcmla_m(p0, svAcc10, b0, a1, 0);
			svAcc10 = svcmla_m(p0, svAcc10, b0, a1, 270);
		}

		if (nr > 1)
		{
			svAcc01 = svcmla_m(p0, svAcc01, b1, a0, 0);
			svAcc01 = svcmla_m(p0, svAcc01, b1, a0, 270);

			if (mr > 1)
			{
				svAcc11 = svcmla_m(p0, svAcc11, b1, a1, 0);
				svAcc11 = svcmla_m(p0, svAcc11, b1, a1, 270);
			}
		}
	}

	if (update != 0)
	{
		C[0] += svaddv_f32(r_active, svAcc00);
		C[1] += svaddv_f32(i_active, svAcc00);
		if (nr > 1)
		{
			C[2] += svaddv_f32(r_active, svAcc01);
			C[3] += svaddv_f32(i_active, svAcc01);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			C[0] += svaddv_f32(r_active, svAcc10);
			C[1] += svaddv_f32(i_active, svAcc10);
			if (nr > 1)
			{
				C[2] += svaddv_f32(r_active, svAcc11);
				C[3] += svaddv_f32(i_active, svAcc11);
			}
		}
	}
	else
	{
		C[0] = svaddv_f32(r_active, svAcc00);
		C[1] = svaddv_f32(i_active, svAcc00);
		if (nr > 1)
		{
			C[2] = svaddv_f32(r_active, svAcc01);
			C[3] = svaddv_f32(i_active, svAcc01);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			C[0] = svaddv_f32(r_active, svAcc10);
			C[1] = svaddv_f32(i_active, svAcc10);
			if (nr > 1)
			{
				C[2] = svaddv_f32(r_active, svAcc11);
				C[3] = svaddv_f32(i_active, svAcc11);
			}
		}
	}
}



//-- old -----------------------------------------------------------------------

void nnp_c4gemm_conjb_only_2x2__neon_old(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float32x4_t acc00r = vdupq_n_f32(0.0f), acc00i = vdupq_n_f32(0.0f);
	float32x4_t acc01r = vdupq_n_f32(0.0f), acc01i = vdupq_n_f32(0.0f);
	float32x4_t acc10r = vdupq_n_f32(0.0f), acc10i = vdupq_n_f32(0.0f);
	float32x4_t acc11r = vdupq_n_f32(0.0f), acc11i = vdupq_n_f32(0.0f);
	do
	{
		const float32x4_t a0r = vld1q_f32_aligned(a + 0);
		const float32x4_t a0i = vld1q_f32_aligned(a + 4);
		const float32x4_t a1r = vld1q_f32_aligned(a + 8);
		const float32x4_t a1i = vld1q_f32_aligned(a + 12);

		const float32x4_t b0r = vld1q_f32_aligned(b + 0);
		const float32x4_t b0i = vld1q_f32_aligned(b + 4);
		const float32x4_t b1r = vld1q_f32_aligned(b + 8);
		const float32x4_t b1i = vld1q_f32_aligned(b + 12);
		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);
		acc01r = vmuladdq_f32(acc01r, a0r, b1r);
		acc01i = vmuladdq_f32(acc01i, a0i, b1r);
		acc11r = vmuladdq_f32(acc11r, a1r, b1r);
		acc11i = vmuladdq_f32(acc11i, a1i, b1r);

		acc00r = vmuladdq_f32(acc00r, a0i, b0i);
		acc00i = vmulsubq_f32(acc00i, a0r, b0i);
		acc10r = vmuladdq_f32(acc10r, a1i, b0i);
		acc10i = vmulsubq_f32(acc10i, a1r, b0i);
		acc01r = vmuladdq_f32(acc01r, a0i, b1i);
		acc01i = vmulsubq_f32(acc01i, a0r, b1i);
		acc11r = vmuladdq_f32(acc11r, a1i, b1i);
		acc11i = vmulsubq_f32(acc11i, a1r, b1i);

		a += 16;
		b += 16;
	} while (--k);

	if (update != 0)
	{
		vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc00r));
		vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc00i));
		vst1q_f32_aligned(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), acc01r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc01i));
		c += row_stride_c;
		vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc10r));
		vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc10i));
		vst1q_f32_aligned(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), acc11r));
		vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
	}
	else
	{
		vst1q_f32_aligned(c + 0, acc00r);
		vst1q_f32_aligned(c + 4, acc00i);
		vst1q_f32_aligned(c + 8, acc01r);
		vst1q_f32_aligned(c + 12, acc01i);
		c += row_stride_c;
		vst1q_f32_aligned(c + 0, acc10r);
		vst1q_f32_aligned(c + 4, acc10i);
		vst1q_f32_aligned(c + 8, acc11r);
		vst1q_f32_aligned(c + 12, acc11i);
	}
}

void nnp_c4gemm_conjb_upto_2x2__neon_old(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	float32x4_t acc00r = vdupq_n_f32(0.0f), acc00i = vdupq_n_f32(0.0f);
	float32x4_t acc01r = vdupq_n_f32(0.0f), acc01i = vdupq_n_f32(0.0f);
	float32x4_t acc10r = vdupq_n_f32(0.0f), acc10i = vdupq_n_f32(0.0f);
	float32x4_t acc11r = vdupq_n_f32(0.0f), acc11i = vdupq_n_f32(0.0f);
	do
	{
		float32x4_t a0r, a0i, a1r, a1i;

		a0r = vld1q_f32_aligned(a + 0);
		a0i = vld1q_f32_aligned(a + 4);
		a += 8;
		if (mr > 1)
		{
			a1r = vld1q_f32_aligned(a + 0);
			a1i = vld1q_f32_aligned(a + 4);
			a += 8;
		}

		const float32x4_t b0r = vld1q_f32_aligned(b + 0);
		const float32x4_t b0i = vld1q_f32_aligned(b + 4);
		b += 8;

		acc00r = vmuladdq_f32(acc00r, a0r, b0r);
		acc00i = vmuladdq_f32(acc00i, a0i, b0r);
		acc10r = vmuladdq_f32(acc10r, a1r, b0r);
		acc10i = vmuladdq_f32(acc10i, a1i, b0r);

		acc00r = vmuladdq_f32(acc00r, a0i, b0i);
		acc00i = vmulsubq_f32(acc00i, a0r, b0i);
		acc10r = vmuladdq_f32(acc10r, a1i, b0i);
		acc10i = vmulsubq_f32(acc10i, a1r, b0i);

		if (nr > 1)
		{
			const float32x4_t b1r = vld1q_f32_aligned(b + 0);
			const float32x4_t b1i = vld1q_f32_aligned(b + 4);
			b += 8;

			acc01r = vmuladdq_f32(acc01r, a0r, b1r);
			acc01i = vmuladdq_f32(acc01i, a0i, b1r);
			acc11r = vmuladdq_f32(acc11r, a1r, b1r);
			acc11i = vmuladdq_f32(acc11i, a1i, b1r);

			acc01r = vmuladdq_f32(acc01r, a0i, b1i);
			acc01i = vmulsubq_f32(acc01i, a0r, b1i);
			acc11r = vmuladdq_f32(acc11r, a1i, b1i);
			acc11i = vmulsubq_f32(acc11i, a1r, b1i);
		}
	} while (--k);

	if (update != 0)
	{
		vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc00r));
		vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc00i));
		if (nr > 1)
		{
			vst1q_f32_aligned(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), acc01r));
			vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc01i));
		}
		if (mr > 1)
		{
			c += row_stride_c;
			vst1q_f32_aligned(c + 0, vaddq_f32(vld1q_f32_aligned(c + 0), acc10r));
			vst1q_f32_aligned(c + 4, vaddq_f32(vld1q_f32_aligned(c + 4), acc10i));
			if (nr > 1)
			{
				vst1q_f32_aligned(c + 8, vaddq_f32(vld1q_f32_aligned(c + 8), acc11r));
				vst1q_f32_aligned(c + 12, vaddq_f32(vld1q_f32_aligned(c + 12), acc11i));
			}
		}
	}
	else
	{
		vst1q_f32_aligned(c + 0, acc00r);
		vst1q_f32_aligned(c + 4, acc00i);
		if (nr > 1)
		{
			vst1q_f32_aligned(c + 8, acc01r);
			vst1q_f32_aligned(c + 12, acc01i);
		}
		if (mr > 1)
		{
			c += row_stride_c;
			vst1q_f32_aligned(c + 0, acc10r);
			vst1q_f32_aligned(c + 4, acc10i);
			if (nr > 1)
			{
				vst1q_f32_aligned(c + 8, acc11r);
				vst1q_f32_aligned(c + 12, acc11i);
			}
		}
	}
}


void nnp_cgemm_conjb_only_2x2__scalar(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	svfloat32_t svAcc00,svAcc01,svAcc10,svAcc11;
	svfloat32_t a0,a1,b0,b1;
	svbool_t p0;
	svAcc00 = svdup_f32(0.0f);
	svAcc01 = svdup_f32(0.0f);
	svAcc10 = svdup_f32(0.0f);
	svAcc11 = svdup_f32(0.0f);


	const svbool_t all_active = svptrue_b32();
	const svbool_t r_active = svdupq_b32(1,0,1,0);
	const svbool_t i_active = svdupq_b32(0,1,0,1);
	
	const svint32_t ind = svzip1(svindex_s32(0,4*4),svindex_s32(1*4,4*4));


	uint64_t numVals = svlen(svAcc00);

	for(uint32_t i = 0; i < k; i +=numVals/2){
	
		p0 = svwhilelt_b32_s32(i*2, k*2);


		a0 = svld1_gather_offset(p0, &A[4*i+0], ind);
		a1 = svld1_gather_offset(p0, &A[4*i+2], ind);


		b0 = svld1_gather_offset(p0, &B[4*i+0], ind);
		b1 = svld1_gather_offset(p0, &B[4*i+2], ind);


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

void nnp_c4gemm_conjb_only_2x2__neon(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	//nnp_cVLgemm_conjb_only_2x2__sve(k, update, A, B, C, row_stride_c);
	//nnp_c4gemm_conjb_only_2x2__neon_old(k, update, A, B, C, row_stride_c);
	//nnp_cgemm_conjb_only_2x2__scalar(k, update, A, B, C, row_stride_c);
	//print_array_f(C,8*4);
	/*
	printf("k: %i \n",k);
	//print_array_cf(A,4);
	//print_array_cf(B,4);

	printf("row: %i \n",row_stride_c);

	printf("update: %i \n",update);

	//nnp_c4gemm_conjb_only_2x2__neon_new(k, update, A, B, C, row_stride_c);
	
	printf("star\n");


	//nnp_c4gemm_conjb_only_2x2__neon_new(k, update, A, B, C, row_stride_c);
	nnp_cVLgemm_conjb_only_2x2__sve(k, update, A, B, C, row_stride_c);


	print_array_f(C,16);
	print_array_f(C + row_stride_c,16);


	printf("old\n");

	nnp_c4gemm_conjb_only_2x2__neon_old(k, update, A, B, C, row_stride_c);
	//nnp_cgemm_conjb_only_2x2__scalar(k, update, A, B, C, row_stride_c);

	print_array_f(C,16);
	print_array_f(C + row_stride_c,16);
	printf("end\n");
	
	exit(0.0);
	//*/
	

}



void nnp_c4gemm_conjb_upto_2x2__neon(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{

	nnp_c4gemm_conjb_upto_2x2__neon_old(mr, nr, k, update, A, B, C, row_stride_c);
	/*
	printf("star\n");

	C[0] = 0;
	C[1] = 0;
	C[2] = 0;
	C[3] = 0;
	C[4] = 0;
	C[5] = 0;
	C[6] = 0;
	C[7] = 0;

	nnp_c4gemm_conjb_upto_2x2__neon_old(mr, nr, k, update, A, B, C, row_stride_c);


	print_array_f(C,8);

	C[0] = 0;
	C[1] = 0;
	C[2] = 0;
	C[3] = 0;
	C[4] = 0;
	C[5] = 0;
	C[6] = 0;
	C[7] = 0;

	printf("new\n");
	nnp_c4gemm_conjb_upto_2x2__neon_new(mr, nr, k, update, A, B, C, row_stride_c);

	print_array_f(C,8);
	printf("end\n");
	*/
}
