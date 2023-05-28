#pragma once

#include <nnpack/fft-constants.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <riscvv/fft/fft-util.h>
#include <nnpack/hwinfo.h>

#include <riscvv/fft/rv-printf.h>

//#include <epi_rvv.h>



//--FFT8x8-------------------------------------------------------------

static inline void riscvv_fft8x8_complex(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;
    const uint32_t HALF_LENGTH = LENGTH / 2;
    const uint32_t QUARTER_LENGTH = HALF_LENGTH / 2;


	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, QUARTER_LENGTH);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);
    
    __epi_2xf32 twiddle_1_r = dupq_f((float []){COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4}, gvl);
    __epi_2xf32 twiddle_1_i = dupq_f((float []){SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4}, gvl);

    __epi_2xf32 twiddle_2_r = dupq_f((float []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
    __epi_2xf32 twiddle_2_i = dupq_f((float []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);

    __epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;

    __epi_2xi32 ind_zip  = zip_concat_8(gvl);
	__epi_2xi32 ind_even = get_ind_even(gvl);
	__epi_2xi32 ind_odd  = get_ind_odd(ind_even, gvl);
    __epi_2xi32 ind_low  = get_ind_low_BLOCK(BLOCK_SIZE, gvl);
	__epi_2xi32 ind_high = get_ind_high_BLOCK(ind_low, BLOCK_SIZE, gvl);


    __epi_2xi32 ind_load = indexN_address(gvl, 0, 1, 16, 4);
    __epi_2xi32 ind_store = indexN_address(gvl, 0, 2, 16, 4);

     int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);

    for(uint32_t i = 0; i < BLOCK_SIZE/2; i += numVals){
        pg = imin(simd_width, QUARTER_LENGTH - i * 4);
        gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

       
        //load 
        a_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 , ind_load, gvl); 
        a_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE , ind_load, gvl); 
        
        b_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2, ind_load, gvl); 
        b_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2 + BLOCK_SIZE, ind_load, gvl);


        //stage 1
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

		new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_1_r, twiddle_1_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_1_r, twiddle_1_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_low, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_low, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_high, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_high, ind_zip, merge, half, gvl);


        //stage 2
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

		new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_2_r, twiddle_2_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_2_r, twiddle_2_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_even, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_even, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_odd, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_odd, ind_zip, merge, half, gvl);


        //stage 3
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);


        //store
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + 0, new_a_r, ind_store, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + 1, new_a_i, ind_store, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 0, new_b_r, ind_store, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 1, new_b_i, ind_store, gvl);
    }

}



static inline void riscvv_ifft8x8_complex(
    float tf[restrict static 16 * 4])
{

    const uint32_t BLOCK_SIZE = 8;
    const uint32_t LENGTH = BLOCK_SIZE * BLOCK_SIZE;
    const uint32_t HALF_LENGTH = LENGTH / 2;
    const uint32_t QUARTER_LENGTH = HALF_LENGTH / 2;


	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, QUARTER_LENGTH);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

    __epi_2xf32 scaled_twiddle_1_r = dupq_f((float []){0.125f * COS_0PI_OVER_4, 0.125f * COS_1PI_OVER_4, 0.125f * COS_2PI_OVER_4, 0.125f * COS_3PI_OVER_4}, gvl);
    __epi_2xf32 scaled_twiddle_1_i = dupq_f((float []){0.125f * SIN_0PI_OVER_4, 0.125f * SIN_1PI_OVER_4, 0.125f * SIN_2PI_OVER_4, 0.125f * SIN_3PI_OVER_4}, gvl);

    __epi_2xf32 twiddle_2_r = dupq_f((float []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
    __epi_2xf32 twiddle_2_i = dupq_f((float []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);
    __epi_2xf32 scale = __builtin_epi_vfmv_v_f_2xf32(0.125f, gvl);

    __epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;

    __epi_2xi32 ind_zip_concat  = zip_concat_8(gvl);
    __epi_2xi32 ind_zip_interleave  = zip_interleave_8(gvl);

    __epi_2xi32 ind_low  = get_ind_low_BLOCK(BLOCK_SIZE, gvl);
	__epi_2xi32 ind_high = get_ind_high_BLOCK(ind_low, BLOCK_SIZE, gvl);


    __epi_2xi32 ind_index = indexN_address(gvl,0, 1, 8, 4);

    int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);

    for(uint32_t i = 0; i < BLOCK_SIZE/2; i += numVals){
        pg = imin(simd_width, QUARTER_LENGTH - i * 4);
        gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

        //load
        a_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + 0, ind_index, gvl); 
        a_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + HALF_LENGTH, ind_index, gvl); 
        
        b_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + 0, ind_index, gvl); 
        b_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + HALF_LENGTH, ind_index, gvl);

        //stage 3
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

        //stage 2
        a_r = shuffle(new_a_r, new_b_r, ind_low, ind_zip_interleave, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_low, ind_zip_interleave, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_high, ind_zip_interleave, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_high, ind_zip_interleave, merge, half, gvl);

	    new_bt_r = mul_twiddle_r(b_r, b_i, twiddle_2_r, twiddle_2_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, twiddle_2_r, twiddle_2_i, gvl);

        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        //stage 1
        a_r = shuffle(new_a_r, new_b_r, ind_low, ind_zip_concat, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_low, ind_zip_concat, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_high, ind_zip_concat, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_high, ind_zip_concat, merge, half, gvl);

        new_bt_r = mul_twiddle_r(b_r, b_i, scaled_twiddle_1_r, scaled_twiddle_1_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, scaled_twiddle_1_r, scaled_twiddle_1_i, gvl);

        a_r = __builtin_epi_vfmul_2xf32(a_r, scale, gvl); 
        a_i = __builtin_epi_vfmul_2xf32(a_i, scale, gvl); 
        
        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        //store
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + 0, new_a_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + HALF_LENGTH , new_a_i, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + 0 , new_b_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + HALF_LENGTH, new_b_i, ind_index, gvl);

    }
}

//--FFT16x16--------------------------------------------------------------





inline static void riscvv_fft16x16_complex(
    float tf[restrict static 16 * 16])
{

    const int BLOCK_SIZE = 16;
    const int LENGTH = BLOCK_SIZE * BLOCK_SIZE;
    const int HALF_LENGTH = LENGTH/2;
    const int QUARTER_LENGTH = HALF_LENGTH/2;
  
    const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
    const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, QUARTER_LENGTH);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

    __epi_2xf32 twiddle_1_r = dupn_f((float []){COS_0PI_OVER_8, COS_1PI_OVER_8, COS_2PI_OVER_8, COS_3PI_OVER_8, COS_4PI_OVER_8, COS_5PI_OVER_8, COS_6PI_OVER_8, COS_7PI_OVER_8}, 8, gvl);
    __epi_2xf32 twiddle_1_i = dupn_f((float []){SIN_0PI_OVER_8, SIN_1PI_OVER_8, SIN_2PI_OVER_8, SIN_3PI_OVER_8, SIN_4PI_OVER_8, SIN_5PI_OVER_8, SIN_6PI_OVER_8, SIN_7PI_OVER_8}, 8, gvl);

    __epi_2xf32 twiddle_2_r = dupq_f((float []){COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4}, gvl);
    __epi_2xf32 twiddle_2_i = dupq_f((float []){SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4}, gvl);

    __epi_2xf32 twiddle_3_r = dupq_f((float []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
    __epi_2xf32 twiddle_3_i = dupq_f((float []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);

    __epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;

    __epi_2xi32 ind_duo_0 = get_ind_even(gvl);
    __epi_2xi32 ind_duo_1 = get_ind_odd(ind_duo_0, gvl);
    __epi_2xi32 ind_tetra_0 = get_ind_tetra_0(gvl);
    __epi_2xi32 ind_tetra_1 = get_ind_tetra_1(ind_tetra_0,gvl);
    __epi_2xi32 ind_octo_0 = get_ind_low_BLOCK(BLOCK_SIZE, gvl);
    __epi_2xi32 ind_octo_1 = get_ind_high_BLOCK(ind_octo_0, BLOCK_SIZE,gvl);

    __epi_2xi32 ind_zip  = zip_concat_16(gvl);

    __epi_2xi32 ind_index = indexN_address(gvl,0, 1, 16*2, 8);

     int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);

    for(int i =0; i < BLOCK_SIZE/2; i += numVals){

        //pg = imin(simd_width, QUARTER_LENGTH - i * 8);
        //gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

       
        //load 
        a_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + 0, ind_index, gvl); 
        a_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE , ind_index, gvl); 
        
        b_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2, ind_index, gvl); 
        b_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE / 2 + BLOCK_SIZE, ind_index, gvl);


        //stage 1
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

		new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_1_r, twiddle_1_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_1_r, twiddle_1_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_octo_0, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_octo_0, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_octo_1, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_octo_1, ind_zip, merge, half, gvl);


        //stage 2
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

		new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_2_r, twiddle_2_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_2_r, twiddle_2_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_tetra_0, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_tetra_0, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_tetra_1, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_tetra_1, ind_zip, merge, half, gvl);


        //stage 3
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

        new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_3_r, twiddle_3_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_3_r, twiddle_3_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_duo_0, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_duo_0, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_duo_1, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_duo_1, ind_zip, merge, half, gvl);

        //stage 4
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);


        //store
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + 0, new_a_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE, new_a_i, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE/2 + 0, new_b_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE * 2 + BLOCK_SIZE/2 + BLOCK_SIZE, new_b_i, ind_index, gvl);
    
    }

}



static inline void riscvv_ifft16x16_complex(
    float tf[restrict static 16 * 16])
{
    const int BLOCK_SIZE = 16;
    const int LENGTH = BLOCK_SIZE * BLOCK_SIZE;
    const int HALF_LENGTH = LENGTH/2;
    const int QUARTER_LENGTH = HALF_LENGTH/2;
  
    const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
    const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, QUARTER_LENGTH);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

    __epi_2xf32 scaled_twiddle_1_r = dupn_f((float []){COS_0PI_OVER_8 * 0.0625f, COS_1PI_OVER_8 * 0.0625f, COS_2PI_OVER_8 * 0.0625f, COS_3PI_OVER_8 * 0.0625f, COS_4PI_OVER_8 * 0.0625f, COS_5PI_OVER_8 * 0.0625f, COS_6PI_OVER_8 * 0.0625f, COS_7PI_OVER_8 * 0.0625f}, 8, gvl);
    __epi_2xf32 scaled_twiddle_1_i = dupn_f((float []){SIN_0PI_OVER_8 * 0.0625f, SIN_1PI_OVER_8 * 0.0625f, SIN_2PI_OVER_8 * 0.0625f, SIN_3PI_OVER_8 * 0.0625f, SIN_4PI_OVER_8 * 0.0625f, SIN_5PI_OVER_8 * 0.0625f, SIN_6PI_OVER_8 * 0.0625f, SIN_7PI_OVER_8 * 0.0625f}, 8, gvl);

    __epi_2xf32 twiddle_2_r = dupq_f((float []){COS_0PI_OVER_4, COS_1PI_OVER_4, COS_2PI_OVER_4, COS_3PI_OVER_4}, gvl);
    __epi_2xf32 twiddle_2_i = dupq_f((float []){SIN_0PI_OVER_4, SIN_1PI_OVER_4, SIN_2PI_OVER_4, SIN_3PI_OVER_4}, gvl);

    __epi_2xf32 twiddle_3_r = dupq_f((float []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
    __epi_2xf32 twiddle_3_i = dupq_f((float []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);

    __epi_2xf32 scale = __builtin_epi_vfmv_v_f_2xf32(0.0625f, gvl);

    __epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;

    __epi_2xi32 ind_octo_0 = get_ind_low_BLOCK(BLOCK_SIZE, gvl);
    __epi_2xi32 ind_octo_1 = get_ind_high_BLOCK(ind_octo_0, BLOCK_SIZE,gvl);

    __epi_2xi32 ind_zip_concat  = zip_concat_16(gvl);
    __epi_2xi32 ind_zip_interleave  = zip_interleave_16(gvl);
    __epi_2xi32 ind_zip_mix = zip_mix_16(gvl);

    __epi_2xi32 ind_index = indexN_address(gvl,0, 1, 16, 8);

     int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);

    for(int i =0; i < BLOCK_SIZE/2; i += numVals){

        //pg = imin(simd_width, QUARTER_LENGTH - i * 8);
        //gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

        //load 
        a_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + 0, ind_index, gvl); 
        a_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + HALF_LENGTH, ind_index, gvl); 
        b_r = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + 0, ind_index, gvl); 
        b_i = __builtin_epi_vload_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + HALF_LENGTH, ind_index, gvl);

        //stage 4
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

        //stage 3
        a_r = shuffle(new_a_r, new_b_r, ind_octo_0, ind_zip_interleave, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_octo_0, ind_zip_interleave, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_octo_1, ind_zip_interleave, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_octo_1, ind_zip_interleave, merge, half, gvl);

	    new_bt_r = mul_twiddle_r(b_r, b_i, twiddle_3_r, twiddle_3_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, twiddle_3_r, twiddle_3_i, gvl);

        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        //stage 2
        a_r = shuffle(new_a_r, new_b_r, ind_octo_0, ind_zip_mix, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_octo_0, ind_zip_mix, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_octo_1, ind_zip_mix, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_octo_1, ind_zip_mix, merge, half, gvl);

	    new_bt_r = mul_twiddle_r(b_r, b_i, twiddle_2_r, twiddle_2_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, twiddle_2_r, twiddle_2_i, gvl);

        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        //stage 1
        a_r = shuffle(new_a_r, new_b_r, ind_octo_0, ind_zip_concat, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_octo_0, ind_zip_concat, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_octo_1, ind_zip_concat, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_octo_1, ind_zip_concat, merge, half, gvl);

	    new_bt_r = mul_twiddle_r(b_r, b_i, scaled_twiddle_1_r, scaled_twiddle_1_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, scaled_twiddle_1_r, scaled_twiddle_1_i, gvl);

        a_r = __builtin_epi_vfmul_2xf32(a_r, scale, gvl); 
        a_i = __builtin_epi_vfmul_2xf32(a_i, scale, gvl); 

        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        //store
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + 0, new_a_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + HALF_LENGTH, new_a_i, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + 0, new_b_r, ind_index, gvl);
        __builtin_epi_vstore_indexed_2xf32(tf + i * BLOCK_SIZE + BLOCK_SIZE/2 + HALF_LENGTH, new_b_i, ind_index, gvl);
    
    }
}