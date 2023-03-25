	.arch armv8-a+sve
	.file	"tmp.c"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC1:
	.string	"hej"
	.align	3
.LC2:
	.string	"8x8:"
	.align	3
.LC6:
	.string	"%f "
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 4,,11
	.global	main
	.type	main, %function
main:
.LFB33:
	.cfi_startproc
	sub	sp, sp, #560
	.cfi_def_cfa_offset 560
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	stp	x29, x30, [sp]
	.cfi_offset 29, -560
	.cfi_offset 30, -552
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	str	x21, [sp, 32]
	.cfi_offset 19, -544
	.cfi_offset 20, -536
	.cfi_offset 21, -528
	bl	puts
	mov	x2, 256
	add	x0, sp, 48
	adrp	x1, .LANCHOR0
	add	x1, x1, :lo12:.LANCHOR0
	bl	memcpy
	mov	x2, 256
	mov	w1, 0
	add	x0, sp, 304
	bl	memset
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	puts
	mov	w1, 64
	mov	w2, 40
	index	z2.s, #12, w1
	index	z0.s, #4, w1
	zip1	z0.s, z0.s, z2.s
	index	z1.s, #8, w1
	index	z18.s, #0, w1
	zip1	z18.s, z18.s, z1.s
	zip1	z18.s, z18.s, z0.s
	index	z0.s, w2, w1
	mov	w2, 36
	index	z2.s, w2, w1
	mov	w2, 44
	index	z3.s, w2, w1
	zip1	z2.s, z2.s, z3.s
	mov	w6, 32
	index	z1.s, w6, w1
	mov	w1, 16
	zip1	z1.s, z1.s, z0.s
	index	z0.s, #8, w6
	zip1	z1.s, z1.s, z2.s
	index	z2.s, w1, w6
	zip1	z2.s, z0.s, z2.s
	index	z7.s, #5, #8
	adrp	x1, .LC3
	index	z17.s, #0, w6
	add	x1, x1, :lo12:.LC3
	zip1	z17.s, z17.s, z0.s
	zip1	z18.s, z18.s, z1.s
	index	z0.s, #12, w6
	index	z1.s, #4, w6
	zip1	z0.s, z1.s, z0.s
	zip1	z17.s, z17.s, z0.s
	zip1	z0.s, z0.s, z2.s
	index	z2.s, #4, #8
	zip1	z5.s, z2.s, z7.s
	ptrue	p0.b, all
	index	z3.s, #0, #8
	index	z16.s, #1, #8
	ld1rqw	z20.s, p0/z, [x1]
	zip1	z4.s, z3.s, z16.s
	adrp	x1, .LC4
	zip1	z4.s, z4.s, z5.s
	add	x1, x1, :lo12:.LC4
	index	z5.s, #6, #8
	zip1	z17.s, z17.s, z0.s
	mov	w0, 0
	ld1rqw	z0.s, p0/z, [x1]
	index	z1.s, #3, #8
	index	z6.s, #7, #8
	zip1	z20.s, z20.s, z0.s
	zip1	z22.s, z5.s, z6.s
	index	z0.s, #2, #8
	zip1	z19.s, z3.s, z0.s
	zip1	z21.s, z0.s, z1.s
	zip1	z3.s, z3.s, z2.s
	zip1	z21.s, z21.s, z22.s
	zip1	z2.s, z2.s, z5.s
	zip1	z0.s, z0.s, z5.s
	zip1	z5.s, z16.s, z1.s
	adrp	x1, .LC5
	zip1	z16.s, z16.s, z7.s
	add	x1, x1, :lo12:.LC5
	zip1	z7.s, z7.s, z6.s
	zip1	z4.s, z4.s, z21.s
	zip1	z19.s, z19.s, z5.s
	zip1	z6.s, z1.s, z6.s
	zip1	z16.s, z3.s, z16.s
	zip1	z7.s, z2.s, z7.s
	zip1	z6.s, z0.s, z6.s
	mov	z5.s, #0
	ld1rqw	z21.s, p0/z, [x1]
	.p2align 3,,7
.L2:
	lsr	w5, w0, 1
	ubfiz	x1, x0, 2, 32
	whilelt	p0.s, w5, w6
	add	x4, x1, 16
	add	w2, w5, 4
	ubfiz	x3, x5, 2, 31
	add	x5, sp, 48
	zip1	p1.s, p0.s, p0.s
	add	x1, x5, x1
	add	x4, x5, x4
	ld1w	z0.s, p1/z, [x1, z18.s, uxtw]
	ld1w	z1.s, p1/z, [x4, z18.s, uxtw]
	movprfx	z3, z0
	fsub	z3.s, p0/m, z3.s, z1.s
	movprfx	z2, z0
	fadd	z2.s, p0/m, z2.s, z1.s
	add	x1, sp, 304
	tbl	z0.s, z2.s, z19.s
	movprfx	z1, z5
	fcmla	z1.s, p0/m, z20.s, z3.s, #0
	tbl	z2.s, z2.s, z7.s
	fcmla	z1.s, p0/m, z20.s, z3.s, #270
	tbl	z3.s, z1.s, z19.s
	tbl	z1.s, z1.s, z7.s
	zip1	z1.s, z2.s, z1.s
	tbl	z1.s, z1.s, z4.s
	add	x1, x1, x3
	zip1	z0.s, z0.s, z3.s
	tbl	z0.s, z0.s, z4.s
	movprfx	z2, z0
	fadd	z2.s, p0/m, z2.s, z1.s
	fsub	z0.s, p0/m, z0.s, z1.s
	tbl	z1.s, z2.s, z16.s
	tbl	z2.s, z2.s, z6.s
	movprfx	z3, z5
	fcmla	z3.s, p0/m, z21.s, z0.s, #0
	fcmla	z3.s, p0/m, z21.s, z0.s, #270
	tbl	z0.s, z3.s, z16.s
	tbl	z3.s, z3.s, z6.s
	zip1	z0.s, z1.s, z0.s
	zip1	z1.s, z2.s, z3.s
	tbl	z0.s, z0.s, z4.s
	tbl	z1.s, z1.s, z4.s
	movprfx	z2, z0
	fadd	z2.s, p0/m, z2.s, z1.s
	st1w	z2.s, p1, [x1, z17.s, uxtw]
	add	x1, sp, 304
	fsub	z0.s, p0/m, z0.s, z1.s
	add	x1, x1, x2, lsl 2
	inch	x0
	st1w	z0.s, p1, [x1, z17.s, uxtw]
	cmp	w0, 63
	bls	.L2
	adrp	x20, .LC6
	add	x19, sp, 368
	add	x20, x20, :lo12:.LC6
	add	x21, sp, 560
	.p2align 3,,7
.L3:
	ldr	s0, [x19], 4
	mov	x0, x20
	fcvt	d0, s0
	bl	printf
	cmp	x21, x19
	bne	.L3
	mov	w0, 10
	bl	putchar
	mov	w0, 0
	ldp	x29, x30, [sp]
	ldp	x19, x20, [sp, 16]
	ldr	x21, [sp, 32]
	add	sp, sp, 560
	.cfi_restore 29
	.cfi_restore 30
	.cfi_restore 21
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE33:
	.size	main, .-main
	.section	.rodata.cst16,"aM",@progbits,16
	.align	4
.LC3:
	.word	1065353216
	.word	1060439283
	.word	0
	.word	-1087044365
	.align	4
.LC4:
	.word	0
	.word	1060439283
	.word	1065353216
	.word	1060439283
	.align	4
.LC5:
	.word	1065353216
	.word	0
	.word	0
	.word	1065353216
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
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
	.ident	"GCC: (GNU Toolchain for the A-profile Architecture 10.2-2020.11 (arm-10.16)) 10.2.1 20201103"
	.section	.note.GNU-stack,"",@progbits
