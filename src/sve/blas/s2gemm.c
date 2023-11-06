#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <nnpack/hwinfo.h>
#include <sve/fft/fft-util.h>
#include <sve/fft/sve-print.h>



void nnp_sVLc2gemm_conjb_only_2x2_Split__sve(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
	uint32_t max_simd_width)
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

	const size_t simd_width = imin(nnp_hwinfo.simd_width, max_simd_width); 
	const svbool_t all = svptrue_b32();
	uint32_t num_real_values = 2;
	svbool_t p0, p2;

	const size_t split_channels = imax(nnp_hwinfo.simd_width / max_simd_width, 1);
	const svuint32_t ind_store = indexN(all, 0, 1, simd_width * 4, simd_width);
	
	const svbool_t p_real = svcmplt(all, indexN(all, 0, 1, 0, simd_width), num_real_values);

	for (int i = 0; i < idiv_ceil(k, split_channels); i++)
	{

		p0 = svwhilelt_b32_s32(i * simd_width * split_channels, k * simd_width);
		p2 = svnand_z(all, p0, p_real);

		a0r = svld1_gather_index(p0, A + i * simd_width * 4 * split_channels + simd_width * 0, ind_store);
		a0i = svld1_gather_index(p0, A + i * simd_width * 4 * split_channels + simd_width * 1, ind_store);

		a1r = svld1_gather_index(p0, A + i * simd_width * 4 * split_channels + simd_width * 2, ind_store);
		a1i = svld1_gather_index(p0, A + i * simd_width * 4 * split_channels + simd_width * 3, ind_store);

		b0r = svld1_gather_index(p0, B + i * simd_width * 4 * split_channels + simd_width * 0, ind_store);
		b0i = svld1_gather_index(p0, B + i * simd_width * 4 * split_channels + simd_width * 1, ind_store);

		b1r = svld1_gather_index(p0, B + i * simd_width * 4 * split_channels + simd_width * 2, ind_store);
		b1i = svld1_gather_index(p0, B + i * simd_width * 4 * split_channels + simd_width * 3, ind_store);

		svAcc00r = svmla_m(p0, svAcc00r, a0r, b0r);
		svAcc00r = svmla_m(p2, svAcc00r, a0i, b0i);

		svAcc01r = svmla_m(p0, svAcc01r, a0r, b1r);
		svAcc01r = svmla_m(p2, svAcc01r, a0i, b1i);

		svAcc10r = svmla_m(p0, svAcc10r, a1r, b0r);
		svAcc10r = svmla_m(p2, svAcc10r, a1i, b0i);

		svAcc11r = svmla_m(p0, svAcc11r, a1r, b1r);
		svAcc11r = svmla_m(p2, svAcc11r, a1i, b1i);



		b0r = svsel(p2, b0r, b0i);
		b1r = svsel(p2, b1r, b1i);



		svAcc00i = svmla_m(p0, svAcc00i, a0i, b0r);
		svAcc00i = svmls_m(p2, svAcc00i, a0r, b0i);

		svAcc01i = svmla_m(p0, svAcc01i, a0i, b1r);
		svAcc01i = svmls_m(p2, svAcc01i, a0r, b1i);

		svAcc10i = svmla_m(p0, svAcc10i, a1i, b0r);
		svAcc10i = svmls_m(p2, svAcc10i, a1r, b0i);

		svAcc11i = svmla_m(p0, svAcc11i, a1i, b1r);
		svAcc11i = svmls_m(p2, svAcc11i, a1r, b1i);
	}

	const svbool_t p_store = svwhilelt_b32_s32(0, simd_width);

	sumSplit(p_store, &svAcc00r, split_channels, simd_width);
	sumSplit(p_store, &svAcc01r, split_channels, simd_width);
	sumSplit(p_store, &svAcc10r, split_channels, simd_width);
	sumSplit(p_store, &svAcc11r, split_channels, simd_width);
	
	sumSplit(p_store, &svAcc00i, split_channels, simd_width);
	sumSplit(p_store, &svAcc01i, split_channels, simd_width);
	sumSplit(p_store, &svAcc10i, split_channels, simd_width);
	sumSplit(p_store, &svAcc11i, split_channels, simd_width);

	if (update != 0)
	{

		svst1(p_store, C + simd_width * 0, svadd_m(p_store, svAcc00r, svld1(p_store, C + simd_width * 0)));
		svst1(p_store, C + simd_width * 1, svadd_m(p_store, svAcc00i, svld1(p_store, C + simd_width * 1)));
		svst1(p_store, C + simd_width * 2, svadd_m(p_store, svAcc01r, svld1(p_store, C + simd_width * 2)));
		svst1(p_store, C + simd_width * 3, svadd_m(p_store, svAcc01i, svld1(p_store, C + simd_width * 3)));
		C += row_stride_c;
		svst1(p_store, C + simd_width * 0, svadd_m(p_store, svAcc10r, svld1(p_store, C + simd_width * 0)));
		svst1(p_store, C + simd_width * 1, svadd_m(p_store, svAcc10i, svld1(p_store, C + simd_width * 1)));
		svst1(p_store, C + simd_width * 2, svadd_m(p_store, svAcc11r, svld1(p_store, C + simd_width * 2)));
		svst1(p_store, C + simd_width * 3, svadd_m(p_store, svAcc11i, svld1(p_store, C + simd_width * 3)));
	}
	else
	{

		svst1(p_store, C + simd_width * 0, svAcc00r);
		svst1(p_store, C + simd_width * 1, svAcc00i);
		svst1(p_store, C + simd_width * 2, svAcc01r);
		svst1(p_store, C + simd_width * 3, svAcc01i);
		C += row_stride_c;
		svst1(p_store, C + simd_width * 0, svAcc10r);
		svst1(p_store, C + simd_width * 1, svAcc10i);
		svst1(p_store, C + simd_width * 2, svAcc11r);
		svst1(p_store, C + simd_width * 3, svAcc11i);
	}
}



void nnp_sVLc2gemm_conjb_upto_2x2_Split__sve(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
	uint32_t max_simd_width)
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

	const size_t simd_width = imin(nnp_hwinfo.simd_width, max_simd_width); 
	const svbool_t all = svptrue_b32();

	uint32_t num_real_values = 2;


	int lenA = mr * 2; // 2 for complex
	int lenB = nr * 2; 
	svbool_t p0, p2;

	const size_t split_channels = imax(nnp_hwinfo.simd_width / max_simd_width, 1);
	const svuint32_t ind_storeA = indexN(all,0,1, simd_width * lenA, simd_width);
	const svuint32_t ind_storeB = indexN(all,0,1, simd_width * lenB, simd_width);
	
	const svbool_t p_real = svcmplt(all, indexN(all, 0, 1, 0, simd_width), num_real_values);

	for (int i = 0; i < idiv_ceil(k, split_channels); i++)
	{
		p0 = svwhilelt_b32_s32(i * simd_width * split_channels, k * simd_width);
		p2 = svnand_z(all, p0, p_real);

		a0r = svld1_gather_index(p0, A + i * simd_width * lenA * split_channels + simd_width * 0, ind_storeA);
		a0i = svld1_gather_index(p0, A + i * simd_width * lenA * split_channels + simd_width * 1, ind_storeA);

		b0r = svld1_gather_index(p0, B + i * simd_width * lenB * split_channels + simd_width * 0, ind_storeB);
		b0i = svld1_gather_index(p0, B + i * simd_width * lenB * split_channels + simd_width * 1, ind_storeB);
		

		svAcc00r = svmla_m(p0, svAcc00r, a0r, b0r);
		svAcc00r = svmla_m(p2, svAcc00r, a0i, b0i);
		

		if (nr > 1)
		{

			b1r = svld1_gather_index(p0, B + i * simd_width * lenB * split_channels + simd_width * 2, ind_storeB);
			b1i = svld1_gather_index(p0, B + i * simd_width * lenB * split_channels + simd_width * 3, ind_storeB);

			
			

			svAcc01r = svmla_m(p0, svAcc01r, a0r, b1r);
			svAcc01r = svmla_m(p2, svAcc01r, a0i, b1i);

		}

		if (mr > 1)
		{

			a1r = svld1_gather_index(p0, A + i * simd_width * lenA * split_channels + simd_width * 2, ind_storeA);
			a1i = svld1_gather_index(p0, A + i * simd_width * lenA * split_channels + simd_width * 3, ind_storeA);

			
			

			svAcc10r = svmla_m(p0, svAcc10r, a1r, b0r);
			svAcc10r = svmla_m(p2, svAcc10r, a1i, b0i);

			if (nr > 1)
			{
				svAcc11r = svmla_m(p0, svAcc11r, a1r, b1r);
				svAcc11r = svmla_m(p2, svAcc11r, a1i, b1i);
			}
		}

		b0r = svsel(p2, b0r, b0i);
		b1r = svsel(p2, b1r, b1i);

		svAcc00i = svmla_m(p0, svAcc00i, a0i, b0r);
		svAcc00i = svmls_m(p2, svAcc00i, a0r, b0i);

		if (nr > 1)
		{

			svAcc01i = svmla_m(p0, svAcc01i, a0i, b1r);
			svAcc01i = svmls_m(p2, svAcc01i, a0r, b1i);
		}

		if (mr > 1)
		{

			svAcc10i = svmla_m(p0, svAcc10i, a1i, b0r);
			svAcc10i = svmls_m(p2, svAcc10i, a1r, b0i);

			if (nr > 1)
			{

				svAcc11i = svmla_m(p0, svAcc11i, a1i, b1r);
				svAcc11i = svmls_m(p2, svAcc11i, a1r, b1i);
			}
		}


	}

	const svbool_t p_store = svwhilelt_b32_s32(0, simd_width); 

	sumSplit(p_store, &svAcc00r, split_channels, simd_width);
	sumSplit(p_store, &svAcc01r, split_channels, simd_width);
	sumSplit(p_store, &svAcc10r, split_channels, simd_width);
	sumSplit(p_store, &svAcc11r, split_channels, simd_width);
	
	sumSplit(p_store, &svAcc00i, split_channels, simd_width);
	sumSplit(p_store, &svAcc01i, split_channels, simd_width);
	sumSplit(p_store, &svAcc10i, split_channels, simd_width);
	sumSplit(p_store, &svAcc11i, split_channels, simd_width);

	if (update != 0)
	{

		svst1(p_store, C + simd_width * 0, svadd_m(p_store, svAcc00r, svld1(p_store, C + simd_width * 0)));
		svst1(p_store, C + simd_width * 1, svadd_m(p_store, svAcc00i, svld1(p_store, C + simd_width * 1)));
		if (nr > 1)
		{
			svst1(p_store, C + simd_width * 2, svadd_m(p_store, svAcc01r, svld1(p_store, C + simd_width * 2)));
			svst1(p_store, C + simd_width * 3, svadd_m(p_store, svAcc01i, svld1(p_store, C + simd_width * 3)));
		}
		if (mr > 1)
		{
			C += row_stride_c;
			svst1(p_store, C + simd_width * 0, svadd_m(p_store, svAcc10r, svld1(p_store, C + simd_width * 0)));
			svst1(p_store, C + simd_width * 1, svadd_m(p_store, svAcc10i, svld1(p_store, C + simd_width * 1)));
			if (nr > 1)
			{
				svst1(p_store, C + simd_width * 2, svadd_m(p_store, svAcc11r, svld1(p_store, C + simd_width * 2)));
				svst1(p_store, C + simd_width * 3, svadd_m(p_store, svAcc11i, svld1(p_store, C + simd_width * 3)));
			}
		}
	}else{

		svst1(p_store, C + simd_width * 0, svAcc00r);
		svst1(p_store, C + simd_width * 1, svAcc00i);
		if (nr > 1)
		{
			svst1(p_store, C + simd_width * 2, svAcc01r);
			svst1(p_store, C + simd_width * 3, svAcc01i);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			svst1(p_store, C + simd_width * 0, svAcc10r);
			svst1(p_store, C + simd_width * 1, svAcc10i);
			if (nr > 1)
			{
				svst1(p_store, C + simd_width * 2, svAcc11r);
				svst1(p_store, C + simd_width * 3, svAcc11i);
			}
		}
	}
}

void nnp_sVLc2gemm_conjb_only_2x2__sve(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
	uint32_t max_simd_width)
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

	const size_t simd_width = imin(nnp_hwinfo.simd_width, max_simd_width); 
	const svbool_t p0 = svwhilelt_b32_s32(0, simd_width);
	const svbool_t all = svptrue_b32();

	uint32_t num_real_values = 2;
	const svbool_t p2 = svnand_z(all, p0, svwhilelt_b32_s32(0, num_real_values));


	for (int i = 0; i < k; i++)
	{

		a0r = svld1(p0, A + i * simd_width * 4 + simd_width * 0);
		a0i = svld1(p0, A + i * simd_width * 4 + simd_width * 1);

		a1r = svld1(p0, A + i * simd_width * 4 + simd_width * 2);
		a1i = svld1(p0, A + i * simd_width * 4 + simd_width * 3);

		b0r = svld1(p0, B + i * simd_width * 4 + simd_width * 0);
		b0i = svld1(p0, B + i * simd_width * 4 + simd_width * 1);

		b1r = svld1(p0, B + i * simd_width * 4 + simd_width * 2);
		b1i = svld1(p0, B + i * simd_width * 4 + simd_width * 3);

		svAcc00r = svmla_m(p0, svAcc00r, a0r, b0r);
		svAcc00r = svmla_m(p2, svAcc00r, a0i, b0i);

		svAcc01r = svmla_m(p0, svAcc01r, a0r, b1r);
		svAcc01r = svmla_m(p2, svAcc01r, a0i, b1i);

		svAcc10r = svmla_m(p0, svAcc10r, a1r, b0r);
		svAcc10r = svmla_m(p2, svAcc10r, a1i, b0i);

		svAcc11r = svmla_m(p0, svAcc11r, a1r, b1r);
		svAcc11r = svmla_m(p2, svAcc11r, a1i, b1i);



		b0r = svsel(p2, b0r, b0i);
		b1r = svsel(p2, b1r, b1i);

		svAcc00i = svmla_m(p0, svAcc00i, a0i, b0r);
		svAcc00i = svmls_m(p2, svAcc00i, a0r, b0i);

		svAcc01i = svmla_m(p0, svAcc01i, a0i, b1r);
		svAcc01i = svmls_m(p2, svAcc01i, a0r, b1i);

		svAcc10i = svmla_m(p0, svAcc10i, a1i, b0r);
		svAcc10i = svmls_m(p2, svAcc10i, a1r, b0i);

		svAcc11i = svmla_m(p0, svAcc11i, a1i, b1r);
		svAcc11i = svmls_m(p2, svAcc11i, a1r, b1i);
	}

	if (update != 0)
	{

		svst1(p0, C + simd_width * 0, svadd_m(p0, svAcc00r, svld1(p0, C + simd_width * 0)));
		svst1(p0, C + simd_width * 1, svadd_m(p0, svAcc00i, svld1(p0, C + simd_width * 1)));
		svst1(p0, C + simd_width * 2, svadd_m(p0, svAcc01r, svld1(p0, C + simd_width * 2)));
		svst1(p0, C + simd_width * 3, svadd_m(p0, svAcc01i, svld1(p0, C + simd_width * 3)));
		C += row_stride_c;
		svst1(p0, C + simd_width * 0, svadd_m(p0, svAcc10r, svld1(p0, C + simd_width * 0)));
		svst1(p0, C + simd_width * 1, svadd_m(p0, svAcc10i, svld1(p0, C + simd_width * 1)));
		svst1(p0, C + simd_width * 2, svadd_m(p0, svAcc11r, svld1(p0, C + simd_width * 2)));
		svst1(p0, C + simd_width * 3, svadd_m(p0, svAcc11i, svld1(p0, C + simd_width * 3)));
	}
	else
	{

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
}



void nnp_sVLc2gemm_conjb_upto_2x2__sve(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
	uint32_t max_simd_width)
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

	const size_t simd_width = imin(nnp_hwinfo.simd_width, max_simd_width); 
	const svbool_t p0 = svwhilelt_b32_s32(0, simd_width); 
	const svbool_t all = svptrue_b32();

	uint32_t num_real_values = 2;

	const svbool_t p2 = svnand_z(all, p0, svwhilelt_b32_s32(0, num_real_values));

	int lenA = mr * 2; // 2 for complex
	int lenB = nr * 2; 

	for (int i = 0; i < k; i++)
	{

		a0r = svld1(p0, A + i * simd_width * lenA + simd_width * 0);
		a0i = svld1(p0, A + i * simd_width * lenA + simd_width * 1);

		b0r = svld1(p0, B + i * simd_width * lenB + simd_width * 0);
		b0i = svld1(p0, B + i * simd_width * lenB + simd_width * 1);
		

		svAcc00r = svmla_m(p0, svAcc00r, a0r, b0r);
		svAcc00r = svmla_m(p2, svAcc00r, a0i, b0i);
		

		if (nr > 1)
		{

			b1r = svld1(p0, B + i * simd_width * lenB + simd_width * 2);
			b1i = svld1(p0, B + i * simd_width * lenB + simd_width * 3);

			
			

			svAcc01r = svmla_m(p0, svAcc01r, a0r, b1r);
			svAcc01r = svmla_m(p2, svAcc01r, a0i, b1i);

		}

		if (mr > 1)
		{

			a1r = svld1(p0, A + i * simd_width * lenA + simd_width * 2);
			a1i = svld1(p0, A + i * simd_width * lenA + simd_width * 3);

			
			

			svAcc10r = svmla_m(p0, svAcc10r, a1r, b0r);
			svAcc10r = svmla_m(p2, svAcc10r, a1i, b0i);

			if (nr > 1)
			{
				svAcc11r = svmla_m(p0, svAcc11r, a1r, b1r);
				svAcc11r = svmla_m(p2, svAcc11r, a1i, b1i);
			}
		}

		//todo use SPLICE
		b0r = svsel(p2, b0r, b0i);
		b1r = svsel(p2, b1r, b1i);

		svAcc00i = svmla_m(p0, svAcc00i, a0i, b0r);
		svAcc00i = svmls_m(p2, svAcc00i, a0r, b0i);

		if (nr > 1)
		{

			svAcc01i = svmla_m(p0, svAcc01i, a0i, b1r);
			svAcc01i = svmls_m(p2, svAcc01i, a0r, b1i);
		}

		if (mr > 1)
		{

			svAcc10i = svmla_m(p0, svAcc10i, a1i, b0r);
			svAcc10i = svmls_m(p2, svAcc10i, a1r, b0i);

			if (nr > 1)
			{

				svAcc11i = svmla_m(p0, svAcc11i, a1i, b1r);
				svAcc11i = svmls_m(p2, svAcc11i, a1r, b1i);
			}
		}


	}

	if (update != 0)
	{

		svst1(p0, C + simd_width * 0, svadd_m(p0, svAcc00r, svld1(p0, C + simd_width * 0)));
		svst1(p0, C + simd_width * 1, svadd_m(p0, svAcc00i, svld1(p0, C + simd_width * 1)));
		if (nr > 1)
		{
			svst1(p0, C + simd_width * 2, svadd_m(p0, svAcc01r, svld1(p0, C + simd_width * 2)));
			svst1(p0, C + simd_width * 3, svadd_m(p0, svAcc01i, svld1(p0, C + simd_width * 3)));
		}
		if (mr > 1)
		{
			C += row_stride_c;
			svst1(p0, C + simd_width * 0, svadd_m(p0, svAcc10r, svld1(p0, C + simd_width * 0)));
			svst1(p0, C + simd_width * 1, svadd_m(p0, svAcc10i, svld1(p0, C + simd_width * 1)));
			if (nr > 1)
			{
				svst1(p0, C + simd_width * 2, svadd_m(p0, svAcc11r, svld1(p0, C + simd_width * 2)));
				svst1(p0, C + simd_width * 3, svadd_m(p0, svAcc11i, svld1(p0, C + simd_width * 3)));
			}
		}
	}else{

		svst1(p0, C + simd_width * 0, svAcc00r);
		svst1(p0, C + simd_width * 1, svAcc00i);
		if (nr > 1)
		{
			svst1(p0, C + simd_width * 2, svAcc01r);
			svst1(p0, C + simd_width * 3, svAcc01i);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			svst1(p0, C + simd_width * 0, svAcc10r);
			svst1(p0, C + simd_width * 1, svAcc10i);
			if (nr > 1)
			{
				svst1(p0, C + simd_width * 2, svAcc11r);
				svst1(p0, C + simd_width * 3, svAcc11i);
			}
		}
	}
}

// for fft8x8
void nnp_s2gemm_only_2x2__sve(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	int max_simd_width = 32;

	if(nnp_hwinfo.simd_width / max_simd_width  < 2){
		nnp_sVLc2gemm_conjb_only_2x2__sve(k, update, a, b, c, row_stride_c, max_simd_width);
	}else{
		nnp_sVLc2gemm_conjb_only_2x2_Split__sve(k, update, a, b, c, row_stride_c, max_simd_width);
	}
	
}

// for fft8x8
void nnp_s2gemm_upto_2x2__sve(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{

	int max_simd_width = 32;

	if(nnp_hwinfo.simd_width / max_simd_width  < 2){
		nnp_sVLc2gemm_conjb_upto_2x2__sve(mr, nr, k, update, a, b, c, row_stride_c, max_simd_width);
	}else{
		nnp_sVLc2gemm_conjb_upto_2x2_Split__sve(mr, nr, k, update, a, b, c, row_stride_c, max_simd_width);
	}
	
}


// for fft16x16
void nnp_s2gemm_only_2x2_2048__sve(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	nnp_sVLc2gemm_conjb_only_2x2__sve(k, update, a, b, c, row_stride_c, 64);	
}

// for fft16x16
void nnp_s2gemm_upto_2x2_2048__sve(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{

	nnp_sVLc2gemm_conjb_upto_2x2__sve(mr, nr, k, update, a, b, c, row_stride_c, 64);
	
	
}

