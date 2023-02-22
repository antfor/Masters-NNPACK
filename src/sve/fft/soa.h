#pragma once

#include <nnpack/fft-constants.h>
#include <scalar/butterfly.h>
#include <arm_sve.h>
#include <stdio.h>

static inline void saxpy_sve(float32_t *x, float32_t *y, float32_t a, uint32_t n)
{
        uint32_t i;
        svbool_t predicate;
        svfloat32_t xseg;
        svfloat32_t yseg;
        uint64_t numVals = svlen_f32(xseg);
        for (i = 0; i < n; i += numVals)
        {
                predicate = svwhilelt_b32_s32(i, n);
                xseg = svld1_f32(predicate, x + i);             // ld1w for x
                yseg = svld1_f32(predicate, y + i);             // ld1w for y
                yseg = svmla_n_f32_m(predicate, yseg, xseg, a); // y+a*x
                svst1_f32(predicate, y + i, yseg);              // st1w for y <-y+a*x
        }
}
static inline void vecprint(float32_t *vec, uint32_t n)
{
        int i;
        for (i = 0; i < n; i++)
        {
                printf("[%f]", vec[i]);
        }
        printf("\n");
}

static inline  void vecset(float32_t *vec, float32_t val, uint32_t n)
{
        int i;
        for (i = 0; i < n; i++)
        {
                vec[i] = val;
        }
}

static inline void test_sve()
{
        uint32_t N = 10;
        float32_t x[N];
        float32_t y[N];
        vecset(x, 1.5, N);
        vecset(y, 2.0, N);
        vecprint(x, N);
        vecprint(y, N);
        // choose implementation
        saxpy_sve(x, y, 2.0, N);
        // saxpy_asm(x, y, 2.0, N);
        vecprint(y, N);
}

static inline void butterfly_sve(uint32_t N, svfloat32_t *pwir, float32_t *indR, float32_t *indI){

	svbool_t p1;
	svfloat32_t wir = *pwir; 
	uint32_t N_DIV_2 = N/2;


	uint64_t numVals = svlen_f32(wir);
	for(uint32_t i=0; i < N; i+=numVals){
		p1 = svwhilelt_b32_s32(i, N); 


		const svfloat32_t svindRS = svld1(p1, &indR[i]);
		const svfloat32_t wrs = svld1_gather_index_f32(p1,wir,svindRS);

		const svfloat32_t svindRE = svld1(p1, &indR[i + N_DIV_2]);
		const svfloat32_t wre = svld1_gather_index_f32(p1,wir,svindRE);
	
	
		const svfloat32_t svindIS = svld1(p1, &indI[i]);
		const svfloat32_t wis = svld1_gather_index_f32(p1,wir,svindIS);

		const svfloat32_t svindIE = svld1(p1, &indI[i + N_DIV_2]);
		const svfloat32_t wie = svld1_gather_index_f32(p1,wir,svindIE);


		svst1_scatter_index(p1, wir, svindRS, svadd_f32_m(p1, wrs, wre));
		svst1_scatter_index(p1, wir, svindRE, svsub_f32_m(p1, wrs, wre));

		svst1_scatter_index(p1, wir, svindIS, svadd_f32_m(p1, wis, wie));
		svst1_scatter_index(p1, wir, svindIE, svsub_f32_m(p1, wis, wie));

	}
}

static inline void twiddlefactor_sve(uint32_t N, svfloat32_t *pwir, float32_t *ind, int conjugate){


	float32_t twIR[8] = {SIN_0PI_OVER_2, COS_0PI_OVER_2,SIN_1PI_OVER_2,COS_1PI_OVER_2 ,SIN_0PI_OVER_2,COS_0PI_OVER_2 ,SIN_1PI_OVER_2,COS_1PI_OVER_2};
	uint32_t N_DIV_2 = N/2;

	svfloat32_t wir = *pwir;
	svfloat23_t tmp_wir;
	svbool_t p1;

	//svint32_t rot = conjugate ? 270 : 90;
	svint32_t rot = 270;
	uint64_t numVals = svlen_f32(wir);

	for(uint32_t  i=0; i < N; i+=numVals){ // i needs to be even ?
		p1 = svwhilelt_b32_s32(i, N); 

		const svfloat32_t svind = svld1(p1, &ind[i]);
		const svfloat32_t wirD = svld1_gather_index_f32(p1, wir, svind);
		const svfloat32_t cirD = svld1(p1,&twIR[i]);

		svcmla_m(p1,tmp_wir , wirD, cirD, 0);
		svcmla_m(p1,tmp_wir , wirD, cirD, rot); 

		svst1_scatter_index(p1, wir, svind, tmp_wir);
	}
}

static inline void scalar_fft8_soa_new(
	const float t[restrict static 16],
	float f0r[restrict static 1],
	float f1r[restrict static 1],
	float f2r[restrict static 1],
	float f3r[restrict static 1],
	float f4r[restrict static 1],
	float f5r[restrict static 1],
	float f6r[restrict static 1],
	float f7r[restrict static 1],
	float f0i[restrict static 1],
	float f1i[restrict static 1],
	float f2i[restrict static 1],
	float f3i[restrict static 1],
	float f4i[restrict static 1],
	float f5i[restrict static 1],
	float f6i[restrict static 1],
	float f7i[restrict static 1])
{
	printf("sve_fft8_soa\n");
	//test_sve();
	/* Load inputs and FFT8: butterfly */
	 
	/*old load
	float w0r = t[0], w4r = t[4];
	float w1r = t[1], w5r = t[5];
	float w2r = t[2], w6r = t[6];
	float w3r = t[3], w7r = t[7];

	float w0i = t[8], w4i = t[12];
	float w1i = t[9], w5i = t[13];
	float w2i = t[10], w6i = t[14];
	float w3i = t[11], w7i = t[15];
	*/

	uint32_t i;
    svbool_t predicate;
	uint32_t BLOCK_SIZE = 8;
	uint32_t BLOCK_SIZE_DIV_2 = 4;
	svfloat32_t wr0123; //todo make tuple
	svfloat32_t wr4567; //todo make tuple
	svfloat32_t wi0123; //todo make tuple
	svfloat32_t wi4567; //todo make tuple

	uint64_t numVals = svlen_f32(wr0123);
	// load
	for(i = 0; i < BLOCK_SIZE_DIV_2; i+=numVals){

		predicate = svwhilelt_b32_s32(i, BLOCK_SIZE);       
		wr0123 = svld1_f32(predicate, t + i);  
		wr4567 = svld1_f32(predicate, t + i + BLOCK_SIZE_DIV_2);                          // ld1w for x
        wi0123 = svld1_f32(predicate, t + i + BLOCK_SIZE + BLOCK_SIZE_DIV_2);             // ld1w for y
        wi4567 = svld1_f32(predicate, t + i + BLOCK_SIZE + BLOCK_SIZE_DIV_2);
	}

	//old butterfly
	/*
	scalar_butterfly(&w0r, &w4r);
	scalar_butterfly(&w1r, &w5r);
	scalar_butterfly(&w2r, &w6r);
	scalar_butterfly(&w3r, &w7r);

	scalar_butterfly(&w0i, &w4i);
	scalar_butterfly(&w1i, &w5i);
	scalar_butterfly(&w2i, &w6i);
	scalar_butterfly(&w3i, &w7i);
	*/

	svfloat32_t new_wr0123; //todo make tuple
	svfloat32_t new_wr4567; 
	svfloat32_t new_wi0123; //todo make tuple
	svfloat32_t new_wi4567;
	
	// butterfly
	for(i = 0; i < BLOCK_SIZE_DIV_2; i+=numVals){
		predicate = svwhilelt_b32_s32(i, BLOCK_SIZE_DIV_2); 

		new_wr0123 = svadd_f32_m(predicate, wr0123, wr4567);
		new_wr4567 = svsub_f32_m(predicate, wr0123, wr4567);

		new_wi0123 = svadd_f32_m(predicate, wi0123, wi4567);
		new_wi4567 = svsub_f32_m(predicate, wi0123, wi4567);
		
	}

	svbool_t p1,p2;
	svfloat32_t wr;  
	svfloat32_t wi;

	//uint64_t numVals = svlen_f32(wr0123);
	for(i=0; i < BLOCK_SIZE_DIV_2; i+=numVals){
		p1 = svwhilelt_b32_s32(i, BLOCK_SIZE_DIV_2); 
		//p2 = svwhilelt_b32_s32(i + BLOCK_SIZE_DIV_2, BLOCK_SIZE); 

		const svfloat32_t wr0123 =  svld1(p1, &t[i]);
		const svfloat32_t wr4567 =  svld1(p1, &t[i + BLOCK_SIZE_DIV_2]);

		const svfloat32_t wi0123 =  svld1(p1, &t[i + BLOCK_SIZE]);
		const svfloat32_t wi4567 =  svld1(p1, &t[i + BLOCK_SIZE_DIV_2 + BLOCK_SIZE]);

		wr = svadd_f32_m(p1, wr0123, wr4567);
		svst1_scatter_offset(p1, wr, BLOCK_SIZE_DIV_2 ,svsub_f32_m(p1, wr0123, wr4567));

		wi = svadd_f32_m(p1, wi0123, wi4567);
		svst1_scatter_offset(p1, wi, BLOCK_SIZE_DIV_2 ,svsub_f32_m(p1, wi0123, wi4567));
	}

	//twiddle

	float32_t twIR[8] = {SIN_0PI_OVER_2, COS_0PI_OVER_2,SIN_1PI_OVER_2,COS_1PI_OVER_2 ,SIN_0PI_OVER_2,COS_0PI_OVER_2 ,SIN_1PI_OVER_2,COS_1PI_OVER_2};
	svfloat32_t cir;

	svfloat32_t tmpir = svdup_f32(0.0f);
	svfloat32_t wir;
	svint32_t indi, indr;
	
	for(i=0; i < BLOCK_SIZE; i+=numVals){ // i needs to be even ?
		p1 = svwhilelt_b32_s32(i, BLOCK_SIZE); 
		p2 = svwhilelt_b32_s32(i*2.0f, BLOCK_SIZE*2.0f); 

		//load twiddle factors
		cir = svld1(p1,&twIR[i]);

		const svfloat32_t wi4567 = svld1_offset(p1, wi, BLOCK_SIZE_DIV_2);
		indi = svindex_s32(0.0f,2.0f); 
		svst1_scatter_index(p1, wir,indi, wi4567);

		const svfloat32_t wr4567 = svld1_offset(p1, wr, BLOCK_SIZE_DIV_2);
		indr = svindex_s32(1.0f,2.0f); 
		svst1_scatter_index(p1, wir,indr, wr4567);
		
		 
		svcmla_z(p2, tmpir, wir, cir, 0);
		svcmla_z(p2, tmpir, wir, cir, 270);


	}

	/*
	 * 2x FFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w1r, &w3r);
	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly(&w5r, &w7r);

	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly(&w1i, &w3i);
	scalar_butterfly_with_negated_b(&w4i, &w6i);
	scalar_butterfly_with_negated_b(&w5i, &w7i);

	/*
	 * 2x FFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = w3i, -w3r
	 *   w7r, w7i = w7i, -w7r
	 *
	 * (negation of w3i and w7i is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);

	/*
	 * 4x FFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly_with_negated_b(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly_with_negated_b(&w6i, &w7i);

	/* Bit reversal */
	scalar_swap(&w1r, &w4r);
	scalar_swap(&w3r, &w6r);

	scalar_swap(&w1i, &w4i);
	scalar_swap(&w3i, &w6i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
	*f4r = w4r;

	*f4i = w4i;
	*f5r = w5r;
	*f5i = w5i;
	*f6r = w6r;
	*f6i = w6i;
	*f7r = w7r;
	*f7i = w7i;
}

static inline void scalar_fft8_soa_old(
	const float t[restrict static 16],
	float f0r[restrict static 1],
	float f1r[restrict static 1],
	float f2r[restrict static 1],
	float f3r[restrict static 1],
	float f4r[restrict static 1],
	float f5r[restrict static 1],
	float f6r[restrict static 1],
	float f7r[restrict static 1],
	float f0i[restrict static 1],
	float f1i[restrict static 1],
	float f2i[restrict static 1],
	float f3i[restrict static 1],
	float f4i[restrict static 1],
	float f5i[restrict static 1],
	float f6i[restrict static 1],
	float f7i[restrict static 1])
{
	printf("scalar_fft8_soa\n");
	test_sve();
	//test_sve();
	/* Load inputs and FFT8: butterfly */
	float w0r = t[0], w4r = t[4];
	scalar_butterfly(&w0r, &w4r);

	float w1r = t[1], w5r = t[5];
	scalar_butterfly(&w1r, &w5r);

	float w2r = t[2], w6r = t[6];
	scalar_butterfly(&w2r, &w6r);

	float w3r = t[3], w7r = t[7];
	scalar_butterfly(&w3r, &w7r);

	float w0i = t[8], w4i = t[12];
	scalar_butterfly(&w0i, &w4i);

	float w1i = t[9], w5i = t[13];
	scalar_butterfly(&w1i, &w5i);

	float w2i = t[10], w6i = t[14];
	scalar_butterfly(&w2i, &w6i);

	float w3i = t[11], w7i = t[15];
	scalar_butterfly(&w3i, &w7i);

	/*
	 * FFT8: multiplication by twiddle factors:
	 *   
	 *   w5r, w5i = sqrt(2)/2 (w5i + w5r),  sqrt(2)/2 (w5i - w5r)
	 *   w6r, w6i = w6i, -w6r
	 *   w7r, w7i = sqrt(2)/2 (w7i - w7r), -sqrt(2)/2 (w7i + w7r)
	 *
	 * (negation of w6i and w7i is merged into the next butterfly)
	 */
	const float sqrt2_over_2 = SQRT2_OVER_2;
	const float new_w5r = sqrt2_over_2 * (w5i + w5r);
	const float new_w5i = sqrt2_over_2 * (w5i - w5r);
	const float new_w7r = sqrt2_over_2 * (w7i - w7r);
	const float minus_new_w7i = sqrt2_over_2 * (w7i + w7r);
	w5r = new_w5r;
	w5i = new_w5i;
	scalar_swap(&w6r, &w6i);
	w7r = new_w7r;
	w7i = minus_new_w7i;

	/*
	 * 2x FFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);
	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly_with_negated_b(&w4i, &w6i);
	scalar_butterfly(&w5r, &w7r);
	scalar_butterfly_with_negated_b(&w5i, &w7i);

	/*
	 * 2x FFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = w3i, -w3r
	 *   w7r, w7i = w7i, -w7r
	 *
	 * (negation of w3i and w7i is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);

	/*
	 * 4x FFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly_with_negated_b(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly_with_negated_b(&w6i, &w7i);

	/* Bit reversal */
	scalar_swap(&w1r, &w4r);
	scalar_swap(&w1i, &w4i);
	scalar_swap(&w3r, &w6r);
	scalar_swap(&w3i, &w6i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
	*f4r = w4r;
	*f4i = w4i;
	*f5r = w5r;
	*f5i = w5i;
	*f6r = w6r;
	*f6i = w6i;
	*f7r = w7r;
	*f7i = w7i;
}

static inline void scalar_ifft8_soa(
	float w0r, float w1r, float w2r, float w3r, float w4r, float w5r, float w6r, float w7r,
	float w0i, float w1i, float w2i, float w3i, float w4i, float w5i, float w6i, float w7i,
	float t[restrict static 16])
{
	/* Bit reversal */
	scalar_swap(&w1r, &w4r);
	scalar_swap(&w1i, &w4i);
	scalar_swap(&w3r, &w6r);
	scalar_swap(&w3i, &w6i);

	/*
	 * 4x IFFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly(&w6i, &w7i);

	/*
	 * 2x IFFT4: multiplication by twiddle factors:
	 *
	 *   w3r, w3i = -w3i, w3r
	 *   w7r, w7i = -w7i, w7r
	 *
	 * (negation of w3r and w7r is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);

	/*
	 * 2x IFFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly_with_negated_b(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);
	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly(&w4i, &w6i);
	scalar_butterfly_with_negated_b(&w5r, &w7r);
	scalar_butterfly(&w5i, &w7i);

	/*
	 * IFFT8: multiplication by twiddle factors and scaling by 1/8:
	 *
	 *   w5r, w5i =  sqrt(2)/2 (w5r - w5i), sqrt(2)/2 (w5r + w5i)
	 *   w6r, w6i = -w6i, w6r
	 *   w7r, w7i = -sqrt(2)/2 (w7r + w7i), sqrt(2)/2 (w7r - w7i)
	 *
	 * (negation of w6r and w7r is merged into the next butterfly)
	 */
	const float scaled_sqrt2_over_2 = SQRT2_OVER_2 * 0.125f;
	const float new_w5r       = scaled_sqrt2_over_2 * (w5r - w5i);
	const float new_w5i       = scaled_sqrt2_over_2 * (w5r + w5i);
	const float minus_new_w7r = scaled_sqrt2_over_2 * (w7r + w7i);
	const float new_w7i       = scaled_sqrt2_over_2 * (w7r - w7i);
	w5r = new_w5r;
	w5i = new_w5i;
	scalar_swap(&w6r, &w6i);
	w7r = minus_new_w7r;
	w7i = new_w7i;

	/* IFFT8: scaling of remaining coefficients by 1/8 */
	const float scale = 0.125f;
	w0r *= scale;
	w0i *= scale;
	w1r *= scale;
	w1i *= scale;
	w2r *= scale;
	w2i *= scale;
	w3r *= scale;
	w3i *= scale;
	w4r *= scale;
	w4i *= scale;
	w6r *= scale;
	w6i *= scale;

	/* IFFT8: butterfly and store outputs */
	scalar_butterfly(&w0r, &w4r);
	t[0] = w0r;
	t[4] = w4r;

	scalar_butterfly(&w0i, &w4i);
	t[8] = w0i;
	t[12] = w4i;

	scalar_butterfly(&w1r, &w5r);
	t[1] = w1r;
	t[5] = w5r;

	scalar_butterfly(&w1i, &w5i);
	t[9] = w1i;
	t[13] = w5i;

	scalar_butterfly_with_negated_b(&w2r, &w6r);
	t[2] = w2r;
	t[6] = w6r;

	scalar_butterfly(&w2i, &w6i);
	t[10] = w2i;
	t[14] = w6i;

	scalar_butterfly_with_negated_b(&w3r, &w7r);
	t[3] = w3r;
	t[7] = w7r;

	scalar_butterfly(&w3i, &w7i);
	t[11] = w3i;
	t[15] = w7i;
}

static inline void scalar_fft16_soa(
	const float t[restrict static 32],
	float f0r[restrict static 1],
	float f1r[restrict static 1],
	float f2r[restrict static 1],
	float f3r[restrict static 1],
	float f4r[restrict static 1],
	float f5r[restrict static 1],
	float f6r[restrict static 1],
	float f7r[restrict static 1],
	float f8r[restrict static 1],
	float f9r[restrict static 1],
	float f10r[restrict static 1],
	float f11r[restrict static 1],
	float f12r[restrict static 1],
	float f13r[restrict static 1],
	float f14r[restrict static 1],
	float f15r[restrict static 1],
	float f0i[restrict static 1],
	float f1i[restrict static 1],
	float f2i[restrict static 1],
	float f3i[restrict static 1],
	float f4i[restrict static 1],
	float f5i[restrict static 1],
	float f6i[restrict static 1],
	float f7i[restrict static 1],
	float f8i[restrict static 1],
	float f9i[restrict static 1],
	float f10i[restrict static 1],
	float f11i[restrict static 1],
	float f12i[restrict static 1],
	float f13i[restrict static 1],
	float f14i[restrict static 1],
	float f15i[restrict static 1])
{
	/* Load inputs and FFT16: butterfly */
	float w0r = t[0], w8r = t[8];
	scalar_butterfly(&w0r, &w8r);

	float w1r = t[1], w9r = t[9];
	scalar_butterfly(&w1r, &w9r);

	float w2r = t[2], w10r = t[10];
	scalar_butterfly(&w2r, &w10r);

	float w3r = t[3], w11r = t[11];
	scalar_butterfly(&w3r, &w11r);

	float w4r = t[4], w12r = t[12];
	scalar_butterfly(&w4r, &w12r);

	float w5r = t[5], w13r = t[13];
	scalar_butterfly(&w5r, &w13r);

	float w6r = t[6], w14r = t[14];
	scalar_butterfly(&w6r, &w14r);

	float w7r = t[7], w15r = t[15];
	scalar_butterfly(&w7r, &w15r);

	float w0i = t[16], w8i = t[24];
	scalar_butterfly(&w0i, &w8i);

	float w1i = t[17], w9i = t[25];
	scalar_butterfly(&w1i, &w9i);

	float w2i = t[18], w10i = t[26];
	scalar_butterfly(&w2i, &w10i);

	float w3i = t[19], w11i = t[27];
	scalar_butterfly(&w3i, &w11i);

	float w4i = t[20], w12i = t[28];
	scalar_butterfly(&w4i, &w12i);

	float w5i = t[21], w13i = t[29];
	scalar_butterfly(&w5i, &w13i);

	float w6i = t[22], w14i = t[30];
	scalar_butterfly(&w6i, &w14i);

	float w7i = t[23], w15i = t[31];
	scalar_butterfly(&w7i, &w15i);

	/*
	 * FFT16: multiplication by twiddle factors:
	 *
	 *    w9r,  w9i =  w9r * cos( pi/8) +  w9i * cos(3pi/8),    w9i * cos( pi/8) -  w9r * cos(3pi/8)
	 *   w10r, w10i = sqrt(2)/2 (w10i + w10r),  sqrt(2)/2 (w10i - w10r)
	 *   w11r, w11i = w11r * cos(3pi/8) + w11i * cos( pi/8),   w11i * cos(3pi/8) - w11r * cos( pi/8)
	 *   w12r, w12i = w12i, -w12r
	 *   w13r, w13i = w13i * cos( pi/8) - w13r * cos(3pi/8), -(w13r * cos( pi/8) + w13i * cos(3pi/8))
	 *   w14r, w14i = sqrt(2)/2 (w14i - w14r), -sqrt(2)/2 (w14i + w14r)
	 *   w15r, w15i = w15i * cos(3pi/8) - w15r * cos( pi/8), -(w15r * cos(3pi/8) + w15i * cos( pi/8))
	 *
	 * (negation of w12i, w13i, w14i, and w15i is merged into the next butterfly)
	 */
	{
		const float cos_1pi_over_8 = COS_1PI_OVER_8;
		const float cos_3pi_over_8 = COS_3PI_OVER_8;
		const float new_w9r        = w9r * cos_1pi_over_8 +  w9i * cos_3pi_over_8;
		const float new_w9i        = w9i * cos_1pi_over_8 -  w9r * cos_3pi_over_8;
		const float new_w11r       = w11r * cos_3pi_over_8 + w11i * cos_1pi_over_8;
		const float new_w11i       = w11i * cos_3pi_over_8 - w11r * cos_1pi_over_8;
		const float new_w13r       = w13i * cos_1pi_over_8 - w13r * cos_3pi_over_8;
		const float minus_new_w13i = w13r * cos_1pi_over_8 + w13i * cos_3pi_over_8;
		const float new_w15r       = w15i * cos_3pi_over_8 - w15r * cos_1pi_over_8;
		const float minus_new_w15i = w15r * cos_3pi_over_8 + w15i * cos_1pi_over_8;
		w9r = new_w9r;
		w9i = new_w9i;
		w11r = new_w11r;
		w11i = new_w11i;
		w13r = new_w13r;
		w13i = minus_new_w13i;
		w15r = new_w15r;
		w15i = minus_new_w15i;

		scalar_swap(&w12r, &w12i);

		const float sqrt2_over_2 = SQRT2_OVER_2;
		const float new_w10r       = sqrt2_over_2 * (w10i + w10r);
		const float new_w10i       = sqrt2_over_2 * (w10i - w10r);
		const float new_w14r       = sqrt2_over_2 * (w14i - w14r);
		const float minus_new_w14i = sqrt2_over_2 * (w14i + w14r);
		w10r = new_w10r;
		w10i = new_w10i;
		w14r = new_w14r;
		w14i = minus_new_w14i;
	}

	/*
	 * 2x FFT8: butterfly
	 */
	scalar_butterfly(&w0r, &w4r);
	scalar_butterfly(&w0i, &w4i);
	scalar_butterfly(&w1r, &w5r);
	scalar_butterfly(&w1i, &w5i);
	scalar_butterfly(&w2r, &w6r);
	scalar_butterfly(&w2i, &w6i);
	scalar_butterfly(&w3r, &w7r);
	scalar_butterfly(&w3i, &w7i);

	scalar_butterfly(&w8r, &w12r);
	scalar_butterfly_with_negated_b(&w8i, &w12i);
	scalar_butterfly(&w9r, &w13r);
	scalar_butterfly_with_negated_b(&w9i, &w13i);
	scalar_butterfly(&w10r, &w14r);
	scalar_butterfly_with_negated_b(&w10i, &w14i);
	scalar_butterfly(&w11r, &w15r);
	scalar_butterfly_with_negated_b(&w11i, &w15i);

	/*
	 * 2xFFT8: multiplication by twiddle factors:
	 *   
	 *   w5r, w5i = sqrt(2)/2 (w5i + w5r),  sqrt(2)/2 (w5i - w5r)
	 *   w6r, w6i = w6i, -w6r
	 *   w7r, w7i = sqrt(2)/2 (w7i - w7r), -sqrt(2)/2 (w7i + w7r)
	 *
	 *   w13r, w13i = sqrt(2)/2 (w13i + w13r),  sqrt(2)/2 (w13i - w13r)
	 *   w14r, w14i = w14i, -w14r
	 *   w15r, w15i = sqrt(2)/2 (w15i - w15r), -sqrt(2)/2 (w15i + w15r)
	 *
	 * (negation of w6i, w7i, w14i, and w15i is merged into the next butterfly)
	 */
	{
		const float sqrt2_over_2 = SQRT2_OVER_2;
		const float new_w5r       = sqrt2_over_2 * (w5i + w5r);
		const float new_w5i       = sqrt2_over_2 * (w5i - w5r);
		const float new_w7r       = sqrt2_over_2 * (w7i - w7r);
		const float minus_new_w7i = sqrt2_over_2 * (w7i + w7r);
		w5r = new_w5r;
		w5i = new_w5i;
		w7r = new_w7r;
		w7i = minus_new_w7i;

		const float new_w13r       = sqrt2_over_2 * (w13i + w13r);
		const float new_w13i       = sqrt2_over_2 * (w13i - w13r);
		const float new_w15r       = sqrt2_over_2 * (w15i - w15r);
		const float minus_new_w15i = sqrt2_over_2 * (w15i + w15r);
		w13r = new_w13r;
		w13i = new_w13i;
		w15r = new_w15r;
		w15i = minus_new_w15i;

		scalar_swap(&w6r, &w6i);
		scalar_swap(&w14r, &w14i);
	}

	/*
	 * 4x FFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);

	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly_with_negated_b(&w4i, &w6i);
	scalar_butterfly(&w5r, &w7r);
	scalar_butterfly_with_negated_b(&w5i, &w7i);

	scalar_butterfly(&w8r, &w10r);
	scalar_butterfly(&w8i, &w10i);
	scalar_butterfly(&w9r, &w11r);
	scalar_butterfly(&w9i, &w11i);

	scalar_butterfly(&w12r, &w14r);
	scalar_butterfly_with_negated_b(&w12i, &w14i);
	scalar_butterfly(&w13r, &w15r);
	scalar_butterfly_with_negated_b(&w13i, &w15i);

	/*
	 * 4x FFT4: multiplication by twiddle factors:
	 *
	 *    w3r,  w3i =  w3i,  -w3r
	 *    w7r,  w7i =  w7i,  -w7r
	 *   w11r, w11i = w11i, -w11r
	 *   w15r, w15i = w15i, -w15r
	 *
	 * (negation of w3i, w7i, w11i, and w15i is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);
	scalar_swap(&w11r, &w11i);
	scalar_swap(&w15r, &w15i);

	/*
	 * 8x FFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly_with_negated_b(&w2i, &w3i);

	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly_with_negated_b(&w6i, &w7i);

	scalar_butterfly(&w8r, &w9r);
	scalar_butterfly(&w8i, &w9i);
	scalar_butterfly(&w10r, &w11r);
	scalar_butterfly_with_negated_b(&w10i, &w11i);

	scalar_butterfly(&w12r, &w13r);
	scalar_butterfly(&w12i, &w13i);
	scalar_butterfly(&w14r, &w15r);
	scalar_butterfly_with_negated_b(&w14i, &w15i);

	/* Bit reversal */
	scalar_swap(&w1r, &w8r);
	scalar_swap(&w1i, &w8i);
	scalar_swap(&w2r, &w4r);
	scalar_swap(&w2i, &w4i);
	scalar_swap(&w3r, &w12r);
	scalar_swap(&w3i, &w12i);
	scalar_swap(&w5r, &w10r);
	scalar_swap(&w5i, &w10i);
	scalar_swap(&w7r, &w14r);
	scalar_swap(&w7i, &w14i);
	scalar_swap(&w11r, &w13r);
	scalar_swap(&w11i, &w13i);

	*f0r = w0r;
	*f0i = w0i;
	*f1r = w1r;
	*f1i = w1i;
	*f2r = w2r;
	*f2i = w2i;
	*f3r = w3r;
	*f3i = w3i;
	*f4r = w4r;
	*f4i = w4i;
	*f5r = w5r;
	*f5i = w5i;
	*f6r = w6r;
	*f6i = w6i;
	*f7r = w7r;
	*f7i = w7i;
	*f8r = w8r;
	*f8i = w8i;
	*f9r = w9r;
	*f9i = w9i;
	*f10r = w10r;
	*f10i = w10i;
	*f11r = w11r;
	*f11i = w11i;
	*f12r = w12r;
	*f12i = w12i;
	*f13r = w13r;
	*f13i = w13i;
	*f14r = w14r;
	*f14i = w14i;
	*f15r = w15r;
	*f15i = w15i;
}

static inline void scalar_ifft16_soa(
	float w0r, float w1r, float  w2r, float  w3r, float  w4r, float  w5r, float  w6r, float  w7r,
	float w8r, float w9r, float w10r, float w11r, float w12r, float w13r, float w14r, float w15r,
	float w0i, float w1i, float  w2i, float  w3i, float  w4i, float  w5i, float  w6i, float  w7i,
	float w8i, float w9i, float w10i, float w11i, float w12i, float w13i, float w14i, float w15i,
	float t[restrict static 16])
{
	/* Bit reversal */
	scalar_swap(&w1r, &w8r);
	scalar_swap(&w1i, &w8i);
	scalar_swap(&w2r, &w4r);
	scalar_swap(&w2i, &w4i);
	scalar_swap(&w3r, &w12r);
	scalar_swap(&w3i, &w12i);
	scalar_swap(&w5r, &w10r);
	scalar_swap(&w5i, &w10i);
	scalar_swap(&w7r, &w14r);
	scalar_swap(&w7i, &w14i);
	scalar_swap(&w11r, &w13r);
	scalar_swap(&w11i, &w13i);

	/*
	 * 8x IFFT2: butterfly
	 */
	scalar_butterfly(&w0r, &w1r);
	scalar_butterfly(&w0i, &w1i);
	scalar_butterfly(&w2r, &w3r);
	scalar_butterfly(&w2i, &w3i);
	scalar_butterfly(&w4r, &w5r);
	scalar_butterfly(&w4i, &w5i);
	scalar_butterfly(&w6r, &w7r);
	scalar_butterfly(&w6i, &w7i);
	scalar_butterfly(&w8r, &w9r);
	scalar_butterfly(&w8i, &w9i);
	scalar_butterfly(&w10r, &w11r);
	scalar_butterfly(&w10i, &w11i);
	scalar_butterfly(&w12r, &w13r);
	scalar_butterfly(&w12i, &w13i);
	scalar_butterfly(&w14r, &w15r);
	scalar_butterfly(&w14i, &w15i);

	/*
	 * 4x IFFT4: multiplication by twiddle factors:
	 *
	 *    w3r,  w3i =  -w3i,  w3r
	 *    w7r,  w7i =  -w7i,  w7r
	 *   w11r, w11i = -w11i, w11r
	 *   w15r, w15i = -w15i, w15r
	 *
	 * (negation of w3r, w7r, w11r, and w15r is merged into the next butterfly)
	 */
	scalar_swap(&w3r, &w3i);
	scalar_swap(&w7r, &w7i);
	scalar_swap(&w11r, &w11i);
	scalar_swap(&w15r, &w15i);

	/*
	 * 4x IFFT4: butterfly
	 */
	scalar_butterfly(&w0r, &w2r);
	scalar_butterfly(&w0i, &w2i);
	scalar_butterfly_with_negated_b(&w1r, &w3r);
	scalar_butterfly(&w1i, &w3i);

	scalar_butterfly(&w4r, &w6r);
	scalar_butterfly(&w4i, &w6i);
	scalar_butterfly_with_negated_b(&w5r, &w7r);
	scalar_butterfly(&w5i, &w7i);

	scalar_butterfly(&w8r, &w10r);
	scalar_butterfly(&w8i, &w10i);
	scalar_butterfly_with_negated_b(&w9r, &w11r);
	scalar_butterfly(&w9i, &w11i);

	scalar_butterfly(&w12r, &w14r);
	scalar_butterfly(&w12i, &w14i);
	scalar_butterfly_with_negated_b(&w13r, &w15r);
	scalar_butterfly(&w13i, &w15i);

	/*
	 * 2x IFFT8: multiplication by twiddle factors and scaling by 1/8:
	 *
	 *   w5r, w5i =  sqrt(2)/2 (w5r - w5i), sqrt(2)/2 (w5r + w5i)
	 *   w6r, w6i = -w6i, w6r
	 *   w7r, w7i = -sqrt(2)/2 (w7r + w7i), sqrt(2)/2 (w7r - w7i)
	 *
	 *   w13r, w13i =  sqrt(2)/2 (w13r - w13i), sqrt(2)/2 (w13r + w13i)
	 *   w14r, w14i = -w14i, w14r
	 *   w15r, w15i = -sqrt(2)/2 (w15r + w15i), sqrt(2)/2 (w15r - w15i)
	 *
	 * (negation of w6r, w7r, w14r, and w15r is merged into the next butterfly)
	 */
	{
		const float sqrt2_over_2 = SQRT2_OVER_2;
		const float new_w5r       = sqrt2_over_2 * (w5r - w5i);
		const float new_w5i       = sqrt2_over_2 * (w5r + w5i);
		const float minus_new_w7r = sqrt2_over_2 * (w7r + w7i);
		const float new_w7i       = sqrt2_over_2 * (w7r - w7i);
		w5r = new_w5r;
		w5i = new_w5i;
		w7r = minus_new_w7r;
		w7i = new_w7i;

		const float new_w13r       = sqrt2_over_2 * (w13r - w13i);
		const float new_w13i       = sqrt2_over_2 * (w13r + w13i);
		const float minus_new_w15r = sqrt2_over_2 * (w15r + w15i);
		const float new_w15i       = sqrt2_over_2 * (w15r - w15i);
		w13r = new_w13r;
		w13i = new_w13i;
		w15r = minus_new_w15r;
		w15i = new_w15i;

		scalar_swap(&w6r, &w6i);
		scalar_swap(&w14r, &w14i);
	}

	/*
	 * 2x IFFT8: butterfly
	 */
	scalar_butterfly(&w0r, &w4r);
	scalar_butterfly(&w0i, &w4i);
	scalar_butterfly(&w1r, &w5r);
	scalar_butterfly(&w1i, &w5i);
	scalar_butterfly_with_negated_b(&w2r, &w6r);
	scalar_butterfly(&w2i, &w6i);
	scalar_butterfly_with_negated_b(&w3r, &w7r);
	scalar_butterfly(&w3i, &w7i);

	scalar_butterfly(&w8r, &w12r);
	scalar_butterfly(&w8i, &w12i);
	scalar_butterfly(&w9r, &w13r);
	scalar_butterfly(&w9i, &w13i);
	scalar_butterfly_with_negated_b(&w10r, &w14r);
	scalar_butterfly(&w10i, &w14i);
	scalar_butterfly_with_negated_b(&w11r, &w15r);
	scalar_butterfly(&w11i, &w15i);

	/*
	 * IFFT16: multiplication by twiddle factors and scaling by 1/16:
	 *
	 *    w9r,  w9i =    w9r * cos( pi/8) -  w9i * cos(3pi/8),     w9i * cos( pi/8) +  w9r * cos(3pi/8)
	 *   w10r, w10i =  sqrt(2)/2 (w10r - w10i), sqrt(2)/2 (w10r + w10i)
	 *   w11r, w11i =   w11r * cos(3pi/8) - w11i * cos( pi/8),    w11i * cos(3pi/8) + w11r * cos( pi/8)
	 *   w12r, w12i = -w12i, w12r
	 *   w13r, w13i = -(w13i * cos( pi/8) + w13r * cos(3pi/8)), w13r * cos( pi/8) - w13i * cos(3pi/8)
	 *   w14r, w14i = -sqrt(2)/2 (w14r + w14i), sqrt(2)/2 (w14r - w14i)
	 *   w15r, w15i = -(w15i * cos(3pi/8) + w15r * cos( pi/8)), w15r * cos(3pi/8) - w15i * cos( pi/8)
	 *
	 * (negation of w12r, w13r, w14r, and w15r is merged into the next butterfly)
	 */
	{
		const float scaled_sqrt2_over_2 = SQRT2_OVER_2 * 0.0625f;
		const float new_w10r       = scaled_sqrt2_over_2 * (w10r - w10i);
		const float new_w10i       = scaled_sqrt2_over_2 * (w10r + w10i);
		const float minus_new_w14r = scaled_sqrt2_over_2 * (w14r + w14i);
		const float new_w14i       = scaled_sqrt2_over_2 * (w14r - w14i);
		w10r = new_w10r;
		w10i = new_w10i;
		w14r = minus_new_w14r;
		w14i = new_w14i;

		const float scaled_cos_1pi_over_8 = COS_1PI_OVER_8 * 0.0625f;
		const float scaled_cos_3pi_over_8 = COS_3PI_OVER_8 * 0.0625f;
		const float new_w9r        =  w9r * scaled_cos_1pi_over_8 -  w9i * scaled_cos_3pi_over_8;
		const float new_w9i        =  w9i * scaled_cos_1pi_over_8 +  w9r * scaled_cos_3pi_over_8;
		const float new_w11r       = w11r * scaled_cos_3pi_over_8 - w11i * scaled_cos_1pi_over_8;
		const float new_w11i       = w11i * scaled_cos_3pi_over_8 + w11r * scaled_cos_1pi_over_8;
		const float minus_new_w13r = w13i * scaled_cos_1pi_over_8 + w13r * scaled_cos_3pi_over_8;
		const float new_w13i       = w13r * scaled_cos_1pi_over_8 - w13i * scaled_cos_3pi_over_8;
		const float minus_new_w15r = w15i * scaled_cos_3pi_over_8 + w15r * scaled_cos_1pi_over_8;
		const float new_w15i       = w15r * scaled_cos_3pi_over_8 - w15i * scaled_cos_1pi_over_8;
		w9r = new_w9r;
		w9i = new_w9i;
		w11r = new_w11r;
		w11i = new_w11i;
		w13r = minus_new_w13r;
		w13i = new_w13i;
		w15r = minus_new_w15r;
		w15i = new_w15i;

		scalar_swap(&w12r, &w12i);
	}

	/* IFFT16: scaling of remaining coefficients by 1/16 */
	const float scale = 0.0625f;
	w0r *= scale;
	w0i *= scale;
	w1r *= scale;
	w1i *= scale;
	w2r *= scale;
	w2i *= scale;
	w3r *= scale;
	w3i *= scale;
	w4r *= scale;
	w4i *= scale;
	w5r *= scale;
	w5i *= scale;
	w6r *= scale;
	w6i *= scale;
	w7r *= scale;
	w7i *= scale;
	w8r *= scale;
	w8i *= scale;
	w12r *= scale;
	w12i *= scale;

	/* IFFT16: butterfly and store outputs */
	scalar_butterfly(&w0r, &w8r);
	t[0] = w0r;
	t[8] = w8r;

	scalar_butterfly(&w0i, &w8i);
	t[16] = w0i;
	t[24] = w8i;

	scalar_butterfly(&w1r, &w9r);
	t[1] = w1r;
	t[9] = w9r;

	scalar_butterfly(&w1i, &w9i);
	t[17] = w1i;
	t[25] = w9i;

	scalar_butterfly(&w2r, &w10r);
	t[ 2] =  w2r;
	t[10] = w10r;

	scalar_butterfly(&w2i, &w10i);
	t[18] =  w2i;
	t[26] = w10i;

	scalar_butterfly(&w3r, &w11r);
	t[ 3] =  w3r;
	t[11] = w11r;

	scalar_butterfly(&w3i, &w11i);
	t[19] =  w3i;
	t[27] = w11i;

	scalar_butterfly_with_negated_b(&w4r, &w12r);
	t[ 4] =  w4r;
	t[12] = w12r;

	scalar_butterfly(&w4i, &w12i);
	t[20] =  w4i;
	t[28] = w12i;

	scalar_butterfly_with_negated_b(&w5r, &w13r);
	t[ 5] =  w5r;
	t[13] = w13r;

	scalar_butterfly(&w5i, &w13i);
	t[21] =  w5i;
	t[29] = w13i;

	scalar_butterfly_with_negated_b(&w6r, &w14r);
	t[ 6] =  w6r;
	t[14] = w14r;

	scalar_butterfly(&w6i, &w14i);
	t[22] =  w6i;
	t[30] = w14i;

	scalar_butterfly_with_negated_b(&w7r, &w15r);
	t[ 7] =  w7r;
	t[15] = w15r;

	scalar_butterfly(&w7i, &w15i);
	t[23] =  w7i;
	t[31] = w15i;
}
