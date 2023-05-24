#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <riscvv/fft/fft-util.h>


//--Split--------------------------------------------------------------------

void nnp_cVLgemm_conjb_only_2x2_Split__riscvv(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
    int max_simd_width)
{
    //init
	
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
    const size_t simd_width = imin(nnp_hwinfo_simd_width, max_simd_width);
    const size_t split_channels = imax(nnp_hwinfo_simd_width / max_simd_width, 1);

    long gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width), __epi_e32, __epi_m1);

    __epi_2xf32 a0r, a0i, a1r, a1i;
    __epi_2xf32 b0r, b0i, b1r, b1i;

    __epi_2xf32  rvAcc00r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);


    __epi_2xf32  rvAcc00i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);

    const __epi_2xi32 ind_load = indexN_address(gvl, 0, 1, simd_width * 4, simd_width);

   
    int N = idiv_ceil(k, split_channels);
    for(int i = 0; i < N; i++){

        gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width - i * simd_width * split_channels), __epi_e32, __epi_m1);

        //load
        a0r = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * 4 * split_channels + simd_width * 0, ind_load, gvl);
        a0i = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * 4 * split_channels + simd_width * 1, ind_load, gvl);

        a1r = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * 4 * split_channels + simd_width * 2, ind_load, gvl);
        a1i = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * 4 * split_channels + simd_width * 3, ind_load, gvl);


        b0r = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * 4 * split_channels + simd_width * 0, ind_load, gvl);
        b0i = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * 4 * split_channels + simd_width * 1, ind_load, gvl);

        b1r = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * 4 * split_channels + simd_width * 2, ind_load, gvl);
        b1i = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * 4 * split_channels + simd_width * 3, ind_load, gvl);

        
        //mla
        rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0r, b0r, gvl);
        rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0i, b0i, gvl);
        rvAcc00i = __builtin_epi_vfmacc_2xf32(rvAcc00i, a0i, b0r, gvl);
        rvAcc00i = __builtin_epi_vfnmsac_2xf32(rvAcc00i, a0r, b0i, gvl);

        rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0r, b1r, gvl);
        rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0i, b1i, gvl);
        rvAcc01i = __builtin_epi_vfmacc_2xf32(rvAcc01i, a0i, b1r, gvl);
        rvAcc01i = __builtin_epi_vfnmsac_2xf32(rvAcc01i, a0r, b1i, gvl);

        rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1r, b0r, gvl);
        rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1i, b0i, gvl);
        rvAcc10i = __builtin_epi_vfmacc_2xf32(rvAcc10i, a1i, b0r, gvl);
        rvAcc10i = __builtin_epi_vfnmsac_2xf32(rvAcc10i, a1r, b0i, gvl);

        rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1r, b1r, gvl);
        rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1i, b1i, gvl);
        rvAcc11i = __builtin_epi_vfmacc_2xf32(rvAcc11i, a1i, b1r, gvl);
        rvAcc11i = __builtin_epi_vfnmsac_2xf32(rvAcc11i, a1r, b1i, gvl);
    }

    //sum
    if(k>1){
        gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width), __epi_e32, __epi_m1);

        rvAcc00r = sumSplit(rvAcc00r, split_channels, simd_width, gvl);
        rvAcc01r = sumSplit(rvAcc01r, split_channels, simd_width, gvl);
        rvAcc10r = sumSplit(rvAcc10r, split_channels, simd_width, gvl);
        rvAcc11r = sumSplit(rvAcc11r, split_channels, simd_width, gvl);
            
        rvAcc00i = sumSplit(rvAcc00i, split_channels, simd_width, gvl);
        rvAcc01i = sumSplit(rvAcc01i, split_channels, simd_width, gvl);
        rvAcc10i = sumSplit(rvAcc10i, split_channels, simd_width, gvl);
        rvAcc11i = sumSplit(rvAcc11i, split_channels, simd_width, gvl);
    }
    //store
    gvl = __builtin_epi_vsetvl(simd_width, __epi_e32, __epi_m1);

   	if (update != 0)
	{
        __builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc00r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc00i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc01r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc01i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
		C += row_stride_c;
		__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc10r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc10i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc11r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc11i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
   
    }else{
     	__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc00r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc00i, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc01r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc01i, gvl);
		C += row_stride_c;
		__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc10r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc10i, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc11r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc11i, gvl);
    }

}


void nnp_cVLgemm_conjb_upto_2x2_Split__riscvv(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
    int max_simd_width)
{
    //init
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
    const size_t simd_width = imin(nnp_hwinfo_simd_width, max_simd_width);
    const size_t split_channels = imax(nnp_hwinfo_simd_width / max_simd_width, 1);

    long gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width), __epi_e32, __epi_m1);
    
    __epi_2xf32 a0r, a0i, a1r, a1i;
    __epi_2xf32 b0r, b0i, b1r, b1i;

    __epi_2xf32  rvAcc00r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);


    __epi_2xf32  rvAcc00i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);

    int lenA = mr * 2; // 2 for complex
	int lenB = nr * 2; 

    const __epi_2xi32 ind_loadA = indexN_address(gvl, 0, 1, simd_width * lenA, simd_width);
    const __epi_2xi32 ind_loadB = indexN_address(gvl, 0, 1, simd_width * lenB, simd_width);

   
    int N = idiv_ceil(k, split_channels);
    for (int i = 0; i < N; i++)
	{
        gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width - i * simd_width * split_channels), __epi_e32, __epi_m1);

		a0r = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * lenA * split_channels + simd_width * 0, ind_loadA, gvl);
		a0i = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * lenA * split_channels + simd_width * 1, ind_loadA, gvl);

		b0r = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * lenB * split_channels + simd_width * 0, ind_loadB, gvl);
		b0i = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * lenB * split_channels + simd_width * 1, ind_loadB, gvl);

		rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0r, b0r, gvl);
		rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0i, b0i, gvl);
		rvAcc00i = __builtin_epi_vfmacc_2xf32(rvAcc00i, a0i, b0r, gvl);
		rvAcc00i = __builtin_epi_vfnmsac_2xf32(rvAcc00i, a0r, b0i, gvl);

		if (nr > 1)
		{

			b1r = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * lenB * split_channels + simd_width * 2, ind_loadB, gvl);
			b1i = __builtin_epi_vload_indexed_2xf32(B + i * simd_width * lenB * split_channels + simd_width * 3, ind_loadB, gvl);

			rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0r, b1r, gvl);
			rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0i, b1i, gvl);
			rvAcc01i = __builtin_epi_vfmacc_2xf32(rvAcc01i, a0i, b1r, gvl);
			rvAcc01i = __builtin_epi_vfnmsac_2xf32(rvAcc01i, a0r, b1i, gvl);
		}

		if (mr > 1)
		{

			a1r = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * lenA * split_channels + simd_width * 2, ind_loadA, gvl);
			a1i = __builtin_epi_vload_indexed_2xf32(A + i * simd_width * lenA * split_channels + simd_width * 3, ind_loadA, gvl);

			rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1r, b0r, gvl);
			rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1i, b0i, gvl);
			rvAcc10i = __builtin_epi_vfmacc_2xf32(rvAcc10i, a1i, b0r, gvl);
			rvAcc10i = __builtin_epi_vfnmsac_2xf32(rvAcc10i, a1r, b0i, gvl);

			if (nr > 1)
			{

				rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1r, b1r, gvl);
				rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1i, b1i, gvl);
				rvAcc11i = __builtin_epi_vfmacc_2xf32(rvAcc11i, a1i, b1r, gvl);
				rvAcc11i = __builtin_epi_vfnmsac_2xf32(rvAcc11i, a1r, b1i, gvl);
			}
		}
	}

    //sum
    if(k>1){
        gvl = __builtin_epi_vsetvl(imin(split_channels * simd_width, k * simd_width), __epi_e32, __epi_m1);
   
        rvAcc00r = sumSplit(rvAcc00r, split_channels, simd_width, gvl);
        rvAcc01r = sumSplit(rvAcc01r, split_channels, simd_width, gvl);
        rvAcc10r = sumSplit(rvAcc10r, split_channels, simd_width, gvl);
        rvAcc11r = sumSplit(rvAcc11r, split_channels, simd_width, gvl);
            
        rvAcc00i = sumSplit(rvAcc00i, split_channels, simd_width, gvl);
        rvAcc01i = sumSplit(rvAcc01i, split_channels, simd_width, gvl);
        rvAcc10i = sumSplit(rvAcc10i, split_channels, simd_width, gvl);
        rvAcc11i = sumSplit(rvAcc11i, split_channels, simd_width, gvl);
    }

    //store
    gvl = __builtin_epi_vsetvl(simd_width, __epi_e32, __epi_m1);
	if (update != 0)
	{

		__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc00r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc00i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		if (nr > 1)
		{
			__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc01r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc01i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc10r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc10i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
			if (nr > 1)
			{
				__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc11r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
				__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc11i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
			}
		}
	}else{

		__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc00r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc00i, gvl);
		if (nr > 1)
		{
			__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc01r, gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc01i, gvl);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc10r, gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc10i, gvl);
			if (nr > 1)
			{
				__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc11r, gvl);
				__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc11i, gvl);
			}
		}
	}
}

//---------------------------------------------------------------------------
void nnp_cVLgemm_conjb_only_2x2__riscvv(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
    int simd_width)
{

    long gvl = __builtin_epi_vsetvl(simd_width, __epi_e32, __epi_m1);

    __epi_2xf32 a0r, a0i, a1r, a1i;
    __epi_2xf32 b0r, b0i, b1r, b1i;


    __epi_2xf32  rvAcc00r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);


    __epi_2xf32  rvAcc00i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);

    for(int i = 0; i < k; i++){

        //load
        a0r = __builtin_epi_vload_2xf32(A + i * simd_width * 4 + simd_width * 0, gvl);
        a0i = __builtin_epi_vload_2xf32(A + i * simd_width * 4 + simd_width * 1, gvl);

        a1r = __builtin_epi_vload_2xf32(A + i * simd_width * 4 + simd_width * 2, gvl);
        a1i = __builtin_epi_vload_2xf32(A + i * simd_width * 4 + simd_width * 3, gvl);


        b0r = __builtin_epi_vload_2xf32(B + i * simd_width * 4 + simd_width * 0, gvl);
        b0i = __builtin_epi_vload_2xf32(B + i * simd_width * 4 + simd_width * 1, gvl);

        b1r = __builtin_epi_vload_2xf32(B + i * simd_width * 4 + simd_width * 2, gvl);
        b1i = __builtin_epi_vload_2xf32(B + i * simd_width * 4 + simd_width * 3, gvl);

        
        //mla
        rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0r, b0r, gvl);
        rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0i, b0i, gvl);
        rvAcc00i = __builtin_epi_vfmacc_2xf32(rvAcc00i, a0i, b0r, gvl);
        rvAcc00i = __builtin_epi_vfnmsac_2xf32(rvAcc00i, a0r, b0i, gvl);

        rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0r, b1r, gvl);
        rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0i, b1i, gvl);
        rvAcc01i = __builtin_epi_vfmacc_2xf32(rvAcc01i, a0i, b1r, gvl);
        rvAcc01i = __builtin_epi_vfnmsac_2xf32(rvAcc01i, a0r, b1i, gvl);

        rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1r, b0r, gvl);
        rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1i, b0i, gvl);
        rvAcc10i = __builtin_epi_vfmacc_2xf32(rvAcc10i, a1i, b0r, gvl);
        rvAcc10i = __builtin_epi_vfnmsac_2xf32(rvAcc10i, a1r, b0i, gvl);

        rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1r, b1r, gvl);
        rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1i, b1i, gvl);
        rvAcc11i = __builtin_epi_vfmacc_2xf32(rvAcc11i, a1i, b1r, gvl);
        rvAcc11i = __builtin_epi_vfnmsac_2xf32(rvAcc11i, a1r, b1i, gvl);
    }

   	if (update != 0)
	{
        __builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc00r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc00i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc01r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc01i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
		C += row_stride_c;
		__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc10r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc10i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc11r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc11i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
   
    }else{
     	__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc00r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc00i, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc01r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc01i, gvl);
		C += row_stride_c;
		__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc10r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc10i, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc11r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc11i, gvl);
    }

}


void nnp_cVLgemm_conjb_upto_2x2__riscvv(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c,
    int simd_width)
{
    long gvl = __builtin_epi_vsetvl(simd_width, __epi_e32, __epi_m1);
    
    __epi_2xf32 a0r, a0i, a1r, a1i;
    __epi_2xf32 b0r, b0i, b1r, b1i;

    __epi_2xf32  rvAcc00r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11r =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);


    __epi_2xf32  rvAcc00i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc01i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc10i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
    __epi_2xf32  rvAcc11i =  __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);

    int lenA = mr * 2; // 2 for complex
	int lenB = nr * 2; 

    for (int i = 0; i < k; i++)
	{

		a0r = __builtin_epi_vload_2xf32(A + i * simd_width * lenA + simd_width * 0, gvl);
		a0i = __builtin_epi_vload_2xf32(A + i * simd_width * lenA + simd_width * 1, gvl);

		b0r = __builtin_epi_vload_2xf32(B + i * simd_width * lenB + simd_width * 0, gvl);
		b0i = __builtin_epi_vload_2xf32(B + i * simd_width * lenB + simd_width * 1, gvl);

		rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0r, b0r, gvl);
		rvAcc00r = __builtin_epi_vfmacc_2xf32(rvAcc00r, a0i, b0i, gvl);
		rvAcc00i = __builtin_epi_vfmacc_2xf32(rvAcc00i, a0i, b0r, gvl);
		rvAcc00i = __builtin_epi_vfnmsac_2xf32(rvAcc00i, a0r, b0i, gvl);

		if (nr > 1)
		{

			b1r = __builtin_epi_vload_2xf32(B + i * simd_width * lenB + simd_width * 2, gvl);
			b1i = __builtin_epi_vload_2xf32(B + i * simd_width * lenB + simd_width * 3, gvl);

			rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0r, b1r, gvl);
			rvAcc01r = __builtin_epi_vfmacc_2xf32(rvAcc01r, a0i, b1i, gvl);
			rvAcc01i = __builtin_epi_vfmacc_2xf32(rvAcc01i, a0i, b1r, gvl);
			rvAcc01i = __builtin_epi_vfnmsac_2xf32(rvAcc01i, a0r, b1i, gvl);
		}

		if (mr > 1)
		{

			a1r = __builtin_epi_vload_2xf32(A + i * simd_width * lenA + simd_width * 2, gvl);
			a1i = __builtin_epi_vload_2xf32(A + i * simd_width * lenA + simd_width * 3, gvl);

			rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1r, b0r, gvl);
			rvAcc10r = __builtin_epi_vfmacc_2xf32(rvAcc10r, a1i, b0i, gvl);
			rvAcc10i = __builtin_epi_vfmacc_2xf32(rvAcc10i, a1i, b0r, gvl);
			rvAcc10i = __builtin_epi_vfnmsac_2xf32(rvAcc10i, a1r, b0i, gvl);

			if (nr > 1)
			{

				rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1r, b1r, gvl);
				rvAcc11r = __builtin_epi_vfmacc_2xf32(rvAcc11r, a1i, b1i, gvl);
				rvAcc11i = __builtin_epi_vfmacc_2xf32(rvAcc11i, a1i, b1r, gvl);
				rvAcc11i = __builtin_epi_vfnmsac_2xf32(rvAcc11i, a1r, b1i, gvl);
			}
		}
	}

	if (update != 0)
	{

		__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc00r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc00i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
		if (nr > 1)
		{
			__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc01r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc01i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			__builtin_epi_vstore_2xf32(C + simd_width * 0, __builtin_epi_vfadd_2xf32(rvAcc10r, __builtin_epi_vload_2xf32(C + simd_width * 0, gvl), gvl), gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 1, __builtin_epi_vfadd_2xf32(rvAcc10i, __builtin_epi_vload_2xf32(C + simd_width * 1, gvl), gvl), gvl);
			if (nr > 1)
			{
				__builtin_epi_vstore_2xf32(C + simd_width * 2, __builtin_epi_vfadd_2xf32(rvAcc11r, __builtin_epi_vload_2xf32(C + simd_width * 2, gvl), gvl), gvl);
				__builtin_epi_vstore_2xf32(C + simd_width * 3, __builtin_epi_vfadd_2xf32(rvAcc11i, __builtin_epi_vload_2xf32(C + simd_width * 3, gvl), gvl), gvl);
			}
		}
	}else{

		__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc00r, gvl);
		__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc00i, gvl);
		if (nr > 1)
		{
			__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc01r, gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc01i, gvl);
		}
		if (mr > 1)
		{
			C += row_stride_c;
			__builtin_epi_vstore_2xf32(C + simd_width * 0, rvAcc10r, gvl);
			__builtin_epi_vstore_2xf32(C + simd_width * 1, rvAcc10i, gvl);
			if (nr > 1)
			{
				__builtin_epi_vstore_2xf32(C + simd_width * 2, rvAcc11r, gvl);
				__builtin_epi_vstore_2xf32(C + simd_width * 3, rvAcc11i, gvl);
			}
		}
	}
}


//-8x8-------------------------------------------------------------------

void nnp_cgemm_conjb_only_2x2_FFT8x8__riscvv(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{
	int max_simd_width = 32;
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);

	if(nnp_hwinfo_simd_width / max_simd_width  < 2){
		nnp_cVLgemm_conjb_only_2x2__riscvv(k, update, A, B, C, row_stride_c, nnp_hwinfo_simd_width);
	}else{
		nnp_cVLgemm_conjb_only_2x2_Split__riscvv(k, update, A, B, C, row_stride_c, max_simd_width);
	}

}

void nnp_cgemm_conjb_upto_2x2_FFT8x8__riscvv(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	int max_simd_width = 32;
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);

	if(nnp_hwinfo_simd_width / max_simd_width  < 2){
		nnp_cVLgemm_conjb_upto_2x2__riscvv(mr, nr, k, update, A, B, c, row_stride_c, nnp_hwinfo_simd_width);
	}else{
		nnp_cVLgemm_conjb_upto_2x2_Split__riscvv(mr, nr, k, update, A, B, c, row_stride_c, max_simd_width);
	}
}

//--16x16------------------------------------------------------------------

void nnp_cgemm_conjb_only_2x2_FFT16x16__riscvv(
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float C[restrict static 1],
	size_t row_stride_c)
{
	int max_simd_width = 128;
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);

	if(nnp_hwinfo_simd_width / max_simd_width  < 2){
		nnp_cVLgemm_conjb_only_2x2__riscvv(k, update, A, B, C, row_stride_c, nnp_hwinfo_simd_width);
	}else{
		nnp_cVLgemm_conjb_only_2x2_Split__riscvv(k, update, A, B, C, row_stride_c, max_simd_width);
	}

}

void nnp_cgemm_conjb_upto_2x2_FFT16x16__riscvv(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float A[restrict static 1],
	const float B[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c)
{
	int max_simd_width = 128;
	int nnp_hwinfo_simd_width = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
    
	if(nnp_hwinfo_simd_width / max_simd_width  < 2){
		nnp_cVLgemm_conjb_upto_2x2__riscvv(mr, nr, k, update, A, B, c, row_stride_c, nnp_hwinfo_simd_width);
	}else{
		nnp_cVLgemm_conjb_upto_2x2_Split__riscvv(mr, nr, k, update, A, B, c, row_stride_c, max_simd_width);
	}
}