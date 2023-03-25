	.arch armv8-a+sve
	.file	"tmp.c"
	.text
	.align	2
	.type	butterfly, %function
butterfly:
.LFB1:
	.cfi_startproc
	sub	sp, sp, #48
	.cfi_def_cfa_offset 48
	str	x0, [sp, 40]
	str	x1, [sp, 32]
	str	x2, [sp, 24]
	str	x3, [sp, 16]
	str	x4, [sp, 8]
	ldr	x0, [sp, 40]
	ldr	p0, [x0]
	ldr	x0, [sp, 32]
	ptrue	p1.b, all
	ld1w	z0.s, p1/z, [x0]
	ldr	x0, [sp, 24]
	ptrue	p1.b, all
	ld1w	z1.s, p1/z, [x0]
	fadd	z0.s, p0/m, z0.s, z1.s
	ldr	x0, [sp, 16]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	ldr	x0, [sp, 40]
	ldr	p0, [x0]
	ldr	x0, [sp, 32]
	ptrue	p1.b, all
	ld1w	z0.s, p1/z, [x0]
	ldr	x0, [sp, 24]
	ptrue	p1.b, all
	ld1w	z1.s, p1/z, [x0]
	fsub	z0.s, p0/m, z0.s, z1.s
	ldr	x0, [sp, 8]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	nop
	add	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE1:
	.size	butterfly, .-butterfly
	.align	2
	.type	cmul_twiddle, %function
cmul_twiddle:
.LFB3:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, 24]
	str	x1, [sp, 16]
	str	x2, [sp, 8]
	str	x3, [sp]
	mov	z0.s, #0
	ldr	x0, [sp]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	ldr	x0, [sp, 24]
	ldr	p0, [x0]
	ldr	x0, [sp]
	ptrue	p1.b, all
	ld1w	z0.s, p1/z, [x0]
	ldr	x0, [sp, 8]
	ptrue	p1.b, all
	ld1w	z1.s, p1/z, [x0]
	ldr	x0, [sp, 16]
	ptrue	p1.b, all
	ld1w	z2.s, p1/z, [x0]
	fcmla	z0.s, p0/m, z1.s, z2.s, #0
	ldr	x0, [sp]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	ldr	x0, [sp, 24]
	ldr	p0, [x0]
	ldr	x0, [sp]
	ptrue	p1.b, all
	ld1w	z0.s, p1/z, [x0]
	ldr	x0, [sp, 8]
	ptrue	p1.b, all
	ld1w	z1.s, p1/z, [x0]
	ldr	x0, [sp, 16]
	ptrue	p1.b, all
	ld1w	z2.s, p1/z, [x0]
	fcmla	z0.s, p0/m, z1.s, z2.s, #270
	ldr	x0, [sp]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	nop
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE3:
	.size	cmul_twiddle, .-cmul_twiddle
	.align	2
	.type	suffle, %function
suffle:
.LFB4:
	.cfi_startproc
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	str	x0, [sp, 56]
	str	x1, [sp, 48]
	str	x2, [sp, 40]
	str	x3, [sp, 32]
	str	x4, [sp, 24]
	str	x5, [sp, 16]
	str	x6, [sp, 8]
	str	x7, [sp]
	ldr	x0, [sp, 48]
	ptrue	p0.b, all
	ld1w	z0.s, p0/z, [x0]
	ldr	x0, [sp, 32]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	tbl	z0.s, z0.s, z1.s
	ldr	x0, [sp, 40]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	ldr	x0, [sp, 32]
	ptrue	p0.b, all
	ld1w	z2.s, p0/z, [x0]
	tbl	z1.s, z1.s, z2.s
	zip1	z0.s, z0.s, z1.s
	ldr	x0, [sp, 16]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	tbl	z0.s, z0.s, z1.s
	ldr	x0, [sp, 8]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	ldr	x0, [sp, 48]
	ptrue	p0.b, all
	ld1w	z0.s, p0/z, [x0]
	ldr	x0, [sp, 24]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	tbl	z0.s, z0.s, z1.s
	ldr	x0, [sp, 40]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	ldr	x0, [sp, 24]
	ptrue	p0.b, all
	ld1w	z2.s, p0/z, [x0]
	tbl	z1.s, z1.s, z2.s
	zip1	z0.s, z0.s, z1.s
	ldr	x0, [sp, 16]
	ptrue	p0.b, all
	ld1w	z1.s, p0/z, [x0]
	tbl	z0.s, z0.s, z1.s
	ldr	x0, [sp]
	ptrue	p0.b, all
	st1w	z0.s, p0, [x0]
	nop
	add	sp, sp, 64
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4:
	.size	suffle, .-suffle
	.align	2
	.variant_pcs	index2
	.type	index2, %function
index2:
.LFB5:
	.cfi_startproc
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, 12]
	str	w1, [sp, 8]
	str	w2, [sp, 4]
	ldr	w1, [sp, 12]
	ldr	w0, [sp, 4]
	index	z0.s, w1, w0
	ldr	w1, [sp, 8]
	ldr	w0, [sp, 4]
	index	z1.s, w1, w0
	zip1	z0.s, z0.s, z1.s
	add	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE5:
	.size	index2, .-index2
	.align	2
	.variant_pcs	index4
	.type	index4, %function
index4:
.LFB6:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp
	addvl	sp, sp, #-1
	.cfi_escape 0xf,0xa,0x8f,0,0x92,0x2e,0,0x38,0x1e,0x23,0x30,0x22
	str	z8, [sp]
	.cfi_escape 0x10,0x48,0x2,0x8f,0
	addvl	x5, sp, #1
	str	w0, [x5, 44]
	addvl	x0, sp, #1
	str	w1, [x0, 40]
	addvl	x0, sp, #1
	str	w2, [x0, 36]
	addvl	x0, sp, #1
	str	w3, [x0, 32]
	addvl	x0, sp, #1
	str	w4, [x0, 28]
	addvl	x0, sp, #1
	ldr	w2, [x0, 28]
	addvl	x0, sp, #1
	ldr	w1, [x0, 36]
	addvl	x0, sp, #1
	ldr	w0, [x0, 44]
	bl	index2
	mov	z8.d, z0.d
	addvl	x0, sp, #1
	ldr	w2, [x0, 28]
	addvl	x0, sp, #1
	ldr	w1, [x0, 32]
	addvl	x0, sp, #1
	ldr	w0, [x0, 40]
	bl	index2
	zip1	z0.s, z8.s, z0.s
	ldr	z8, [sp]
	addvl	sp, sp, #1
	.cfi_def_cfa_offset 48
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 72
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE6:
	.size	index4, .-index4
	.align	2
	.variant_pcs	index8
	.type	index8, %function
index8:
.LFB7:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp
	addvl	sp, sp, #-1
	.cfi_escape 0xf,0xa,0x8f,0,0x92,0x2e,0,0x38,0x1e,0x23,0x30,0x22
	str	z8, [sp]
	.cfi_escape 0x10,0x48,0x2,0x8f,0
	addvl	x8, sp, #1
	str	w0, [x8, 44]
	addvl	x0, sp, #1
	str	w1, [x0, 40]
	addvl	x0, sp, #1
	str	w2, [x0, 36]
	addvl	x0, sp, #1
	str	w3, [x0, 32]
	addvl	x0, sp, #1
	str	w4, [x0, 28]
	addvl	x0, sp, #1
	str	w5, [x0, 24]
	addvl	x0, sp, #1
	str	w6, [x0, 20]
	addvl	x0, sp, #1
	str	w7, [x0, 16]
	addvl	x0, sp, #1
	ldr	w4, [x0, 48]
	addvl	x0, sp, #1
	ldr	w3, [x0, 20]
	addvl	x0, sp, #1
	ldr	w2, [x0, 28]
	addvl	x0, sp, #1
	ldr	w1, [x0, 36]
	addvl	x0, sp, #1
	ldr	w0, [x0, 44]
	bl	index4
	mov	z8.d, z0.d
	addvl	x0, sp, #1
	ldr	w4, [x0, 48]
	addvl	x0, sp, #1
	ldr	w3, [x0, 16]
	addvl	x0, sp, #1
	ldr	w2, [x0, 24]
	addvl	x0, sp, #1
	ldr	w1, [x0, 32]
	addvl	x0, sp, #1
	ldr	w0, [x0, 40]
	bl	index4
	zip1	z0.s, z8.s, z0.s
	ldr	z8, [sp]
	addvl	sp, sp, #1
	.cfi_def_cfa_offset 48
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 72
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE7:
	.size	index8, .-index8
	.section	.rodata
	.align	3
.LC1:
	.string	"%f "
	.text
	.align	2
	.type	print_array_f, %function
print_array_f:
.LFB13:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp
	str	x0, [sp, 24]
	str	w1, [sp, 20]
	str	wzr, [sp, 44]
	b	.L11
.L12:
	ldrsw	x0, [sp, 44]
	lsl	x0, x0, 2
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	ldr	s0, [x0]
	fcvt	d0, s0
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	bl	printf
	ldr	w0, [sp, 44]
	add	w0, w0, 1
	str	w0, [sp, 44]
.L11:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 20]
	cmp	w1, w0
	blt	.L12
	mov	w0, 10
	bl	putchar
	nop
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE13:
	.size	print_array_f, .-print_array_f
	.align	2
	.type	fft8x8c, %function
fft8x8c:
.LFB14:
	.cfi_startproc
	addvl	sp, sp, #-16
	.cfi_escape 0xf,0x9,0x8f,0,0x92,0x2e,0,0x8,0x80,0x1e,0x22
	sub	sp, sp, #96
	.cfi_escape 0xf,0xb,0x8f,0,0x92,0x2e,0,0x8,0x80,0x1e,0x23,0x60,0x22
	stp	x29, x30, [sp, 16]
	.cfi_escape 0x10,0x1d,0x2,0x8f,0x10
	.cfi_escape 0x10,0x1e,0x2,0x8f,0x18
	add	x29, sp, 16
	str	x0, [sp, 56]
	str	x1, [sp, 48]
	str	x2, [sp, 40]
	mov	w0, 8
	addvl	x1, sp, #16
	str	w0, [x1, 88]
	addvl	x0, sp, #16
	ldr	w0, [x0, 88]
	mul	w0, w0, w0
	addvl	x1, sp, #16
	str	w0, [x1, 84]
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	ptrue	p0.b, all
	ld1rqw	z0.s, p0/z, [x0]
	adrp	x0, .LC3
	add	x0, x0, :lo12:.LC3
	ptrue	p0.b, all
	ld1rqw	z1.s, p0/z, [x0]
	zip1	z0.s, z0.s, z1.s
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #12
	st1w	z0.s, p0, [x0]
	adrp	x0, .LC4
	add	x0, x0, :lo12:.LC4
	ptrue	p0.b, all
	ld1rqw	z0.s, p0/z, [x0]
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #11
	st1w	z0.s, p0, [x0]
	cntw	x0
	lsl	x0, x0, 1
	addvl	x1, sp, #16
	str	x0, [x1, 72]
	mov	w0, 8
	str	w0, [sp]
	mov	w7, 7
	mov	w6, 5
	mov	w5, 3
	mov	w4, 1
	mov	w3, 6
	mov	w2, 4
	mov	w1, 2
	mov	w0, 0
	bl	index8
	ptrue	p0.b, all
	add	x0, sp, 64
	st1w	z0.s, p0, [x0, #4, mul vl]
	mov	w4, 8
	mov	w3, 3
	mov	w2, 2
	mov	w1, 1
	mov	w0, 0
	bl	index4
	ptrue	p0.b, all
	add	x0, sp, 64
	st1w	z0.s, p0, [x0, #3, mul vl]
	mov	w4, 8
	mov	w3, 7
	mov	w2, 6
	mov	w1, 5
	mov	w0, 4
	bl	index4
	ptrue	p0.b, all
	add	x0, sp, 64
	st1w	z0.s, p0, [x0, #2, mul vl]
	mov	w4, 8
	mov	w3, 5
	mov	w2, 4
	mov	w1, 1
	mov	w0, 0
	bl	index4
	ptrue	p0.b, all
	add	x0, sp, 64
	st1w	z0.s, p0, [x0, #1, mul vl]
	mov	w4, 8
	mov	w3, 7
	mov	w2, 6
	mov	w1, 3
	mov	w0, 2
	bl	index4
	ptrue	p0.b, all
	add	x0, sp, 64
	st1w	z0.s, p0, [x0]
	mov	w0, 64
	str	w0, [sp]
	mov	w7, 44
	mov	w6, 12
	mov	w5, 40
	mov	w4, 8
	mov	w3, 36
	mov	w2, 4
	mov	w1, 32
	mov	w0, 0
	bl	index8
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #15
	st1w	z0.s, p0, [x0]
	ldr	x0, [sp, 40]
	lsl	w1, w0, 2
	ldr	x0, [sp, 40]
	add	x0, x0, 1
	lsl	w2, w0, 2
	ldr	x0, [sp, 40]
	lsl	w3, w0, 3
	ldr	x0, [sp, 40]
	lsl	w0, w0, 3
	add	w4, w0, 4
	ldr	x0, [sp, 40]
	mov	w5, w0
	mov	w0, w5
	lsl	w0, w0, 1
	add	w0, w0, w5
	lsl	w0, w0, 2
	mov	w8, w0
	ldr	x0, [sp, 40]
	mov	w5, w0
	mov	w0, w5
	lsl	w0, w0, 1
	add	w0, w0, w5
	lsl	w0, w0, 2
	add	w5, w0, 4
	ldr	x0, [sp, 40]
	mov	w6, w0
	addvl	x0, sp, #16
	ldr	w0, [x0, 88]
	mul	w0, w6, w0
	lsl	w0, w0, 2
	str	w0, [sp]
	mov	w7, w5
	mov	w6, w8
	mov	w5, w4
	mov	w4, w3
	mov	w3, w2
	mov	w2, w1
	mov	w1, 4
	mov	w0, 0
	bl	index8
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #14
	st1w	z0.s, p0, [x0]
	addvl	x0, sp, #16
	str	wzr, [x0, 92]
	b	.L14
.L15:
	addvl	x0, sp, #16
	ldr	w0, [x0, 92]
	lsr	w0, w0, 1
	mov	w1, w0
	addvl	x0, sp, #16
	ldr	w0, [x0, 84]
	lsr	w0, w0, 1
	whilelt	p0.s, w1, w0
	add	x0, sp, 64
	str	p0, [x0, #87, mul vl]
	add	x0, sp, 64
	ldr	p0, [x0, #87, mul vl]
	add	x0, sp, 64
	ldr	p1, [x0, #87, mul vl]
	zip1	p0.s, p0.s, p1.s
	add	x0, sp, 64
	str	p0, [x0, #111, mul vl]
	addvl	x0, sp, #16
	ldr	w0, [x0, 92]
	lsl	x0, x0, 2
	ldr	x1, [sp, 56]
	add	x0, x1, x0
	ptrue	p0.b, all
	add	x1, sp, 64
	incb	x1, all, mul #15
	ld1w	z0.s, p0/z, [x1]
	add	x1, sp, 64
	ldr	p0, [x1, #111, mul vl]
	ld1w	z0.s, p0/z, [x0, z0.s, uxtw]
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #8
	st1w	z0.s, p0, [x0]
	addvl	x0, sp, #16
	ldr	w1, [x0, 92]
	addvl	x0, sp, #16
	ldr	w0, [x0, 88]
	lsr	w0, w0, 1
	uxtw	x0, w0
	add	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 56]
	add	x0, x1, x0
	ptrue	p0.b, all
	add	x1, sp, 64
	incb	x1, all, mul #15
	ld1w	z0.s, p0/z, [x1]
	add	x1, sp, 64
	ldr	p0, [x1, #111, mul vl]
	ld1w	z0.s, p0/z, [x0, z0.s, uxtw]
	ptrue	p0.b, all
	add	x0, sp, 64
	incb	x0, all, mul #9
	st1w	z0.s, p0, [x0]
	addvl	x4, sp, #7
	add	x4, x4, 64
	addvl	x3, sp, #6
	add	x3, x3, 64
	addvl	x2, sp, #9
	add	x2, x2, 64
	addvl	x5, sp, #8
	add	x5, x5, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x5
	bl	butterfly
	addvl	x3, sp, #5
	add	x3, x3, 64
	addvl	x2, sp, #12
	add	x2, x2, 64
	addvl	x4, sp, #7
	add	x4, x4, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x4
	bl	cmul_twiddle
	addvl	x7, sp, #9
	add	x7, x7, 64
	addvl	x6, sp, #8
	add	x6, x6, 64
	addvl	x5, sp, #4
	add	x5, x5, 64
	addvl	x4, sp, #2
	add	x4, x4, 64
	addvl	x3, sp, #3
	add	x3, x3, 64
	addvl	x2, sp, #5
	add	x2, x2, 64
	addvl	x8, sp, #6
	add	x8, x8, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x8
	bl	suffle
	addvl	x4, sp, #7
	add	x4, x4, 64
	addvl	x3, sp, #6
	add	x3, x3, 64
	addvl	x2, sp, #9
	add	x2, x2, 64
	addvl	x5, sp, #8
	add	x5, x5, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x5
	bl	butterfly
	addvl	x3, sp, #5
	add	x3, x3, 64
	addvl	x2, sp, #11
	add	x2, x2, 64
	addvl	x4, sp, #7
	add	x4, x4, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x4
	bl	cmul_twiddle
	addvl	x7, sp, #9
	add	x7, x7, 64
	addvl	x6, sp, #8
	add	x6, x6, 64
	addvl	x5, sp, #4
	add	x5, x5, 64
	add	x4, sp, 64
	addvl	x3, sp, #1
	add	x3, x3, 64
	addvl	x2, sp, #5
	add	x2, x2, 64
	addvl	x8, sp, #6
	add	x8, x8, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x8
	bl	suffle
	addvl	x4, sp, #7
	add	x4, x4, 64
	addvl	x3, sp, #6
	add	x3, x3, 64
	addvl	x2, sp, #9
	add	x2, x2, 64
	addvl	x5, sp, #8
	add	x5, x5, 64
	cntd	x1
	mov	x0, x1
	lsl	x0, x0, 2
	add	x0, x0, x1
	lsl	x0, x0, 3
	add	x0, x0, x1
	neg	x0, x0
	sub	x0, x0, #32
	addvl	x1, sp, #16
	add	x1, x1, 96
	add	x0, x1, x0
	mov	x1, x5
	bl	butterfly
	addvl	x0, sp, #16
	ldr	w0, [x0, 92]
	lsr	w0, w0, 1
	uxtw	x1, w0
	ldr	x0, [sp, 40]
	mul	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 48]
	add	x0, x1, x0
	ptrue	p0.b, all
	add	x1, sp, 64
	ld1w	z1.s, p0/z, [x1, #6, mul vl]
	ptrue	p0.b, all
	add	x1, sp, 64
	incb	x1, all, mul #14
	ld1w	z0.s, p0/z, [x1]
	add	x1, sp, 64
	ldr	p0, [x1, #111, mul vl]
	st1w	z1.s, p0, [x0, z0.s, uxtw]
	addvl	x0, sp, #16
	ldr	w0, [x0, 92]
	lsr	w0, w0, 1
	uxtw	x0, w0
	add	x1, x0, 4
	ldr	x0, [sp, 40]
	mul	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 48]
	add	x0, x1, x0
	ptrue	p0.b, all
	add	x1, sp, 64
	ld1w	z1.s, p0/z, [x1, #7, mul vl]
	ptrue	p0.b, all
	add	x1, sp, 64
	incb	x1, all, mul #14
	ld1w	z0.s, p0/z, [x1]
	add	x1, sp, 64
	ldr	p0, [x1, #111, mul vl]
	st1w	z1.s, p0, [x0, z0.s, uxtw]
	addvl	x0, sp, #16
	ldr	x0, [x0, 72]
	mov	w1, w0
	addvl	x0, sp, #16
	ldr	w0, [x0, 92]
	add	w0, w0, w1
	addvl	x1, sp, #16
	str	w0, [x1, 92]
.L14:
	addvl	x0, sp, #16
	ldr	w1, [x0, 92]
	addvl	x0, sp, #16
	ldr	w0, [x0, 84]
	cmp	w1, w0
	bcc	.L15
	nop
	nop
	ldp	x29, x30, [sp, 16]
	.cfi_restore 29
	.cfi_restore 30
	addvl	sp, sp, #16
	.cfi_def_cfa_offset 96
	add	sp, sp, 96
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE14:
	.size	fft8x8c, .-fft8x8c
	.section	.rodata
	.align	3
.LC5:
	.string	"hej"
	.align	3
.LC6:
	.string	"8x8:"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB22:
	.cfi_startproc
	sub	sp, sp, #592
	.cfi_def_cfa_offset 592
	stp	x29, x30, [sp]
	.cfi_offset 29, -592
	.cfi_offset 30, -584
	mov	x29, sp
	adrp	x0, .LC5
	add	x0, x0, :lo12:.LC5
	bl	puts
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	add	x0, sp, 336
	mov	x3, x1
	mov	x1, 256
	mov	x2, x1
	mov	x1, x3
	bl	memcpy
	add	x0, sp, 80
	mov	x1, 256
	mov	x2, x1
	mov	w1, 0
	bl	memset
	stp	xzr, xzr, [sp, 16]
	stp	xzr, xzr, [sp, 32]
	stp	xzr, xzr, [sp, 48]
	stp	xzr, xzr, [sp, 64]
	adrp	x0, .LC6
	add	x0, x0, :lo12:.LC6
	bl	puts
	add	x1, sp, 80
	add	x0, sp, 336
	mov	x2, 1
	bl	fft8x8c
	add	x0, sp, 80
	add	x0, x0, 64
	mov	w1, 48
	bl	print_array_f
	mov	w0, 0
	ldp	x29, x30, [sp]
	add	sp, sp, 592
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE22:
	.size	main, .-main
	.section	.rodata
	.align	3
.LC0:
	.word	1065353216
	.word	1073741824
	.word	1077936128
	.word	1082130432
	.word	1084227584
	.word	1086324736
	.word	1088421888
	.word	1090519040
	.word	1091567616
	.word	1092616192
	.word	1093664768
	.word	1094713344
	.word	1095761920
	.word	1096810496
	.word	1097859072
	.word	1098907648
	.word	1065353216
	.word	1073741824
	.word	1077936128
	.word	1082130432
	.word	1084227584
	.word	1086324736
	.word	1088421888
	.word	1090519040
	.word	1091567616
	.word	1092616192
	.word	1093664768
	.word	1094713344
	.word	1095761920
	.word	1096810496
	.word	1097859072
	.word	1098907648
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	0
	.word	1065353216
	.word	1073741824
	.word	1077936128
	.word	1082130432
	.word	1084227584
	.word	1086324736
	.word	1088421888
	.word	1090519040
	.word	1091567616
	.word	1092616192
	.word	1093664768
	.word	1094713344
	.word	1095761920
	.word	1096810496
	.word	1097859072
	.word	1098907648
	.text
	.section	.rodata
	.align	3
	.type	t2.0, %object
	.size	t2.0, 64
t2.0:
	.word	1065353216
	.word	1073741824
	.word	1077936128
	.word	1082130432
	.word	1084227584
	.word	1086324736
	.word	1088421888
	.word	1090519040
	.word	1091567616
	.word	1092616192
	.word	1093664768
	.word	1094713344
	.word	1095761920
	.word	1096810496
	.word	1097859072
	.word	1098907648
	.align	4
.LC2:
	.word	1065353216
	.word	1060439283
	.word	0
	.word	-1087044365
	.align	4
.LC3:
	.word	0
	.word	1060439283
	.word	1065353216
	.word	1060439283
	.align	4
.LC4:
	.word	1065353216
	.word	0
	.word	0
	.word	1065353216
	.ident	"GCC: (GNU Toolchain for the A-profile Architecture 10.2-2020.11 (arm-10.16)) 10.2.1 20201103"
	.section	.note.GNU-stack,"",@progbits
