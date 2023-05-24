#include <nnpack/fft-constants.h>
#include <riscvv/fft/aos.h>
#include <riscvv/fft/fft-util.h>
#include <riscvv/fft/complex.h>
#include <nnpack/hwinfo.h>

#include <riscvv/fft/rv-printf.h>



inline static void fft4xNr( 
	const float t_lo[restrict static 1],
	const float t_hi[restrict static 1],
	size_t stride_t,
	uint32_t row_start, uint32_t row_count,
	float f[restrict static 1],
	const uint32_t column_count)
{

	const uint32_t BLOCK_SIZE = 4;
	const uint32_t HALF_BLOCK_LENGTH = BLOCK_SIZE * column_count;
	const uint32_t QUARTER_BLOCK_LENGTH = HALF_BLOCK_LENGTH / 2;

	const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, column_count * BLOCK_SIZE / 2);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 twiddle_r = dupq_f((float []){COS_0PI_OVER_2, COS_1PI_OVER_2, COS_0PI_OVER_2, COS_1PI_OVER_2}, gvl);
	__epi_2xf32 twiddle_i = dupq_f((float []){SIN_0PI_OVER_2, SIN_1PI_OVER_2, SIN_0PI_OVER_2, SIN_1PI_OVER_2}, gvl);

	__epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;


	__epi_2xi32 ind_zip   = zip_concat_4(gvl);
	//for fft4 even same as low
	__epi_2xi32 ind_low   = get_ind_even(gvl);
	__epi_2xi32 ind_high  = get_ind_odd(gvl);

	
	// Offsets and masks
    const uint32_t row_end = row_start + row_count;
    uint32_t a_pred[4] = {row_start <= 0, row_start <= 1, row_start <= 2, row_start <= 3};
	__epi_2xi32 t_lo_offset_r = aos4_offset_r(&a_pred[0], stride_t, gvl);
	__epi_2xi32 t_lo_offset_i = aos4_offset_i(&a_pred[0], stride_t, gvl);

    uint32_t b_pred[4] = {row_start <= 4 && row_end > 4, row_start <= 5 && row_end > 5, row_start <= 6 && row_end > 6, row_start <= 7 && row_end > 7};
	__epi_2xi32 t_hi_offset_r = aos4_offset_r(&b_pred[0], stride_t, gvl);
	__epi_2xi32 t_hi_offset_i = aos4_offset_i(&b_pred[0], stride_t, gvl);

    uint32_t no_jump[8];
	jump_arr(&no_jump[0], &a_pred[0], &b_pred[0], row_count);

	__epi_2xi1 mask_a_r = aos4_mask_a_r(&no_jump[0], &a_pred[0], &b_pred[0], gvl);
	__epi_2xi1 mask_a_i = aos4_mask_a_i(&no_jump[0], &a_pred[0], &b_pred[0], gvl);
	__epi_2xi1 mask_b_r = aos4_mask_b_r(&no_jump[0], &a_pred[0], &b_pred[0], gvl);
	__epi_2xi1 mask_b_i = aos4_mask_b_i(&no_jump[0], &a_pred[0], &b_pred[0], gvl);
    __epi_2xi1 pg_mask;
    __epi_2xi32 ind_store = rvindex_adress(0, 2, gvl);

    int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);
    __epi_2xf32 zero = __builtin_epi_vfmv_v_f_2xf32(0.0f, gvl);
	for (uint32_t i = 0; i < column_count; i+= numVals)
	{	
        int pg = imin(simd_width, column_count * BLOCK_SIZE / 2 - i * BLOCK_SIZE / 2);
		pg_mask = get_pg_mask(pg, gvl);
    
		// load
		a_r = __builtin_epi_vload_indexed_2xf32_mask(zero, t_lo + i, t_lo_offset_r, __builtin_epi_vmand_2xi1(pg_mask, mask_a_r, gvl), gvl);
		a_i = __builtin_epi_vload_indexed_2xf32_mask(zero, t_lo + i, t_lo_offset_i, __builtin_epi_vmand_2xi1(pg_mask, mask_a_i, gvl), gvl);

		b_r = __builtin_epi_vload_indexed_2xf32_mask(zero, t_hi + i, t_hi_offset_r, __builtin_epi_vmand_2xi1(pg_mask, mask_b_r, gvl), gvl);
		b_i = __builtin_epi_vload_indexed_2xf32_mask(zero, t_hi + i, t_hi_offset_i, __builtin_epi_vmand_2xi1(pg_mask, mask_b_i, gvl), gvl);
		
		// stage1
		new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

		new_bt_r = mulc_twiddle_r(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);
		new_bt_i = mulc_twiddle_i(new_b_r, new_b_i, twiddle_r, twiddle_i, gvl);

		a_r = shuffle(new_a_r, new_bt_r, ind_low, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_bt_i, ind_low, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_bt_r, ind_high, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_bt_i, ind_high, ind_zip, merge, half, gvl);


		// stage2
		new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);


		// store
        __builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE + 0, new_a_r, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE + 1, new_a_i, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE + HALF_BLOCK_LENGTH + 0, new_b_r, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE + HALF_BLOCK_LENGTH + 1, new_b_i, ind_store, pg_mask, gvl);
	}
}



static inline void ifft4xNc(
	float w[restrict static 1],	
	float f[restrict static 64],
	uint32_t column_count)
{

	const uint32_t BLOCK_SIZE = 4;
	const uint32_t HALF_BLOCK_LENGTH = column_count * BLOCK_SIZE;
    const uint32_t QUARTER_BLOCK_LENGTH = HALF_BLOCK_LENGTH / 2;

    const uint64_t simd_width =  __builtin_epi_vsetvlmax(__epi_e32, __epi_m1); // nnp_hwinfo.simd_width;
	const int numVals = (simd_width * 2) / BLOCK_SIZE;

    int pg = imin(simd_width, column_count * BLOCK_SIZE / 2);
    long gvl = __builtin_epi_vsetvl(pg, __epi_e32, __epi_m1);

	__epi_2xf32 scaled_twiddle_r = dupq_f((float []){0.25f * COS_0PI_OVER_2, 0.25f * COS_1PI_OVER_2, 0.25f * COS_0PI_OVER_2, 0.25f * COS_1PI_OVER_2}, gvl);
	__epi_2xf32 scaled_twiddle_i = dupq_f((float []){0.25f * SIN_0PI_OVER_2, 0.25f * SIN_1PI_OVER_2, 0.25f * SIN_0PI_OVER_2, 0.25f * SIN_1PI_OVER_2}, gvl);
    __epi_2xf32 scale = __builtin_epi_vfmv_v_f_2xf32(0.25f, gvl);


	__epi_2xf32 a_r, a_i, new_a_r, new_a_i;
	__epi_2xf32 b_r, b_i, new_b_r, new_b_i, new_bt_r, new_bt_i;

    __epi_2xi32 ind_zip   = zip_concat_4(gvl);
	//for fft4 even same as low
	__epi_2xi32 ind_low   = get_ind_even(gvl);
	__epi_2xi32 ind_high  = get_ind_odd(gvl);


    __epi_2xi32 ind_store = indexN_address(gvl, 0, 2, 8, 2);

    int half = gvl/2;
	__epi_2xi1 merge = get_merge(gvl);
    __epi_2xi1 pg_mask;

	for (uint32_t i = 0; i < column_count; i+= numVals)
	{
        int pg = imin(simd_width, column_count * BLOCK_SIZE / 2 - i * BLOCK_SIZE / 2);
		pg_mask = get_pg_mask(pg, gvl);

        // load
        a_r = __builtin_epi_vload_indexed_2xf32_mask(scale, w + i * BLOCK_SIZE * 2 + 0, ind_store, pg_mask, gvl);
		a_i = __builtin_epi_vload_indexed_2xf32_mask(scale, w + i * BLOCK_SIZE * 2 + 1, ind_store, pg_mask, gvl);
		b_r = __builtin_epi_vload_indexed_2xf32_mask(scale, w + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 0, ind_store, pg_mask, gvl);
		b_i = __builtin_epi_vload_indexed_2xf32_mask(scale, w + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 1, ind_store, pg_mask, gvl);

        // stage2
        new_a_r = butterfly_add(a_r, b_r, gvl);
		new_a_i = butterfly_add(a_i, b_i, gvl);
		new_b_r = butterfly_sub(a_r, b_r, gvl);
		new_b_i = butterfly_sub(a_i, b_i, gvl);

        // stage1
        a_r = shuffle(new_a_r, new_b_r, ind_low, ind_zip, merge, half, gvl);
		a_i = shuffle(new_a_i, new_b_i, ind_low, ind_zip, merge, half, gvl);
		b_r = shuffle(new_a_r, new_b_r, ind_high, ind_zip, merge, half, gvl);
		b_i = shuffle(new_a_i, new_b_i, ind_high, ind_zip, merge, half, gvl);

	    new_bt_r = mul_twiddle_r(b_r, b_i, scaled_twiddle_r, scaled_twiddle_i, gvl);
		new_bt_i = mul_twiddle_i(b_r, b_i, scaled_twiddle_r, scaled_twiddle_i, gvl);

        a_r = __builtin_epi_vfmul_2xf32(a_r, scale, gvl); 
        a_i = __builtin_epi_vfmul_2xf32(a_i, scale, gvl); 
        
        new_a_r = butterfly_add(a_r, new_bt_r, gvl);
		new_a_i = butterfly_add(a_i, new_bt_i, gvl);
		new_b_r = butterfly_sub(a_r, new_bt_r, gvl);
		new_b_i = butterfly_sub(a_i, new_bt_i, gvl);

        // store
        __builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE * 2 + 0, new_a_r, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE * 2 + 1, new_a_i, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 0, new_b_r, ind_store, pg_mask, gvl);
		__builtin_epi_vstore_indexed_2xf32_mask(f + i * BLOCK_SIZE * 2 + BLOCK_SIZE + 1, new_b_i, ind_store, pg_mask, gvl);
    }	
}
