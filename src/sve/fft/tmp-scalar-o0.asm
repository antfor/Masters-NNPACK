	.arch armv8-a+sve
	.file	"tmp.c"
	.text
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
	b	.L2
.L3:
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
.L2:
	ldr	w1, [sp, 44]
	ldr	w0, [sp, 20]
	cmp	w1, w0
	blt	.L3
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
	.type	scalar_swap, %function
scalar_swap:
.LFB17:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, 8]
	str	x1, [sp]
	ldr	x0, [sp]
	ldr	s0, [x0]
	str	s0, [sp, 28]
	ldr	x0, [sp, 8]
	ldr	s0, [x0]
	str	s0, [sp, 24]
	ldr	x0, [sp, 8]
	ldr	s0, [sp, 28]
	str	s0, [x0]
	ldr	x0, [sp]
	ldr	s0, [sp, 24]
	str	s0, [x0]
	nop
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE17:
	.size	scalar_swap, .-scalar_swap
	.align	2
	.type	scalar_butterfly, %function
scalar_butterfly:
.LFB18:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, 8]
	str	x1, [sp]
	ldr	x0, [sp, 8]
	ldr	s1, [x0]
	ldr	x0, [sp]
	ldr	s0, [x0]
	fadd	s0, s1, s0
	str	s0, [sp, 28]
	ldr	x0, [sp, 8]
	ldr	s1, [x0]
	ldr	x0, [sp]
	ldr	s0, [x0]
	fsub	s0, s1, s0
	str	s0, [sp, 24]
	ldr	x0, [sp, 8]
	ldr	s0, [sp, 28]
	str	s0, [x0]
	ldr	x0, [sp]
	ldr	s0, [sp, 24]
	str	s0, [x0]
	nop
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE18:
	.size	scalar_butterfly, .-scalar_butterfly
	.align	2
	.type	scalar_butterfly_with_negated_b, %function
scalar_butterfly_with_negated_b:
.LFB20:
	.cfi_startproc
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, 8]
	str	x1, [sp]
	ldr	x0, [sp, 8]
	ldr	s1, [x0]
	ldr	x0, [sp]
	ldr	s0, [x0]
	fsub	s0, s1, s0
	str	s0, [sp, 28]
	ldr	x0, [sp, 8]
	ldr	s1, [x0]
	ldr	x0, [sp]
	ldr	s0, [x0]
	fadd	s0, s1, s0
	str	s0, [sp, 24]
	ldr	x0, [sp, 8]
	ldr	s0, [sp, 28]
	str	s0, [x0]
	ldr	x0, [sp]
	ldr	s0, [sp, 24]
	str	s0, [x0]
	nop
	add	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE20:
	.size	scalar_butterfly_with_negated_b, .-scalar_butterfly_with_negated_b
	.align	2
	.type	scalar_fft8_soa, %function
scalar_fft8_soa:
.LFB21:
	.cfi_startproc
	stp	x29, x30, [sp, -176]!
	.cfi_def_cfa_offset 176
	.cfi_offset 29, -176
	.cfi_offset 30, -168
	mov	x29, sp
	str	x0, [sp, 72]
	str	x1, [sp, 64]
	str	x2, [sp, 56]
	str	x3, [sp, 48]
	str	x4, [sp, 40]
	str	x5, [sp, 32]
	str	x6, [sp, 24]
	str	x7, [sp, 16]
	ldr	x0, [sp, 72]
	ldr	s0, [x0]
	str	s0, [sp, 152]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 16]
	str	s0, [sp, 148]
	add	x1, sp, 148
	add	x0, sp, 152
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 4]
	str	s0, [sp, 144]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 20]
	str	s0, [sp, 140]
	add	x1, sp, 140
	add	x0, sp, 144
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 8]
	str	s0, [sp, 136]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 24]
	str	s0, [sp, 132]
	add	x1, sp, 132
	add	x0, sp, 136
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 12]
	str	s0, [sp, 128]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 28]
	str	s0, [sp, 124]
	add	x1, sp, 124
	add	x0, sp, 128
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 32]
	str	s0, [sp, 120]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 48]
	str	s0, [sp, 116]
	add	x1, sp, 116
	add	x0, sp, 120
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 36]
	str	s0, [sp, 112]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 52]
	str	s0, [sp, 108]
	add	x1, sp, 108
	add	x0, sp, 112
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 40]
	str	s0, [sp, 104]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 56]
	str	s0, [sp, 100]
	add	x1, sp, 100
	add	x0, sp, 104
	bl	scalar_butterfly
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 44]
	str	s0, [sp, 96]
	ldr	x0, [sp, 72]
	ldr	s0, [x0, 60]
	str	s0, [sp, 92]
	add	x1, sp, 92
	add	x0, sp, 96
	bl	scalar_butterfly
	mov	w0, 1267
	movk	w0, 0x3f35, lsl 16
	fmov	s0, w0
	str	s0, [sp, 172]
	ldr	s1, [sp, 108]
	ldr	s0, [sp, 140]
	fadd	s0, s1, s0
	ldr	s1, [sp, 172]
	fmul	s0, s1, s0
	str	s0, [sp, 168]
	ldr	s1, [sp, 108]
	ldr	s0, [sp, 140]
	fsub	s0, s1, s0
	ldr	s1, [sp, 172]
	fmul	s0, s1, s0
	str	s0, [sp, 164]
	ldr	s1, [sp, 92]
	ldr	s0, [sp, 124]
	fsub	s0, s1, s0
	ldr	s1, [sp, 172]
	fmul	s0, s1, s0
	str	s0, [sp, 160]
	ldr	s1, [sp, 92]
	ldr	s0, [sp, 124]
	fadd	s0, s1, s0
	ldr	s1, [sp, 172]
	fmul	s0, s1, s0
	str	s0, [sp, 156]
	ldr	s0, [sp, 168]
	str	s0, [sp, 140]
	ldr	s0, [sp, 164]
	str	s0, [sp, 108]
	add	x1, sp, 100
	add	x0, sp, 132
	bl	scalar_swap
	ldr	s0, [sp, 160]
	str	s0, [sp, 124]
	ldr	s0, [sp, 156]
	str	s0, [sp, 92]
	add	x1, sp, 136
	add	x0, sp, 152
	bl	scalar_butterfly
	add	x1, sp, 104
	add	x0, sp, 120
	bl	scalar_butterfly
	add	x1, sp, 128
	add	x0, sp, 144
	bl	scalar_butterfly
	add	x1, sp, 96
	add	x0, sp, 112
	bl	scalar_butterfly
	add	x1, sp, 132
	add	x0, sp, 148
	bl	scalar_butterfly
	add	x1, sp, 100
	add	x0, sp, 116
	bl	scalar_butterfly_with_negated_b
	add	x1, sp, 124
	add	x0, sp, 140
	bl	scalar_butterfly
	add	x1, sp, 92
	add	x0, sp, 108
	bl	scalar_butterfly_with_negated_b
	add	x1, sp, 96
	add	x0, sp, 128
	bl	scalar_swap
	add	x1, sp, 92
	add	x0, sp, 124
	bl	scalar_swap
	add	x1, sp, 144
	add	x0, sp, 152
	bl	scalar_butterfly
	add	x1, sp, 112
	add	x0, sp, 120
	bl	scalar_butterfly
	add	x1, sp, 128
	add	x0, sp, 136
	bl	scalar_butterfly
	add	x1, sp, 96
	add	x0, sp, 104
	bl	scalar_butterfly_with_negated_b
	add	x1, sp, 140
	add	x0, sp, 148
	bl	scalar_butterfly
	add	x1, sp, 108
	add	x0, sp, 116
	bl	scalar_butterfly
	add	x1, sp, 124
	add	x0, sp, 132
	bl	scalar_butterfly
	add	x1, sp, 92
	add	x0, sp, 100
	bl	scalar_butterfly_with_negated_b
	add	x1, sp, 148
	add	x0, sp, 144
	bl	scalar_swap
	add	x1, sp, 116
	add	x0, sp, 112
	bl	scalar_swap
	add	x1, sp, 132
	add	x0, sp, 128
	bl	scalar_swap
	add	x1, sp, 100
	add	x0, sp, 96
	bl	scalar_swap
	ldr	s0, [sp, 152]
	ldr	x0, [sp, 64]
	str	s0, [x0]
	ldr	s0, [sp, 120]
	ldr	x0, [sp, 184]
	str	s0, [x0]
	ldr	s0, [sp, 144]
	ldr	x0, [sp, 56]
	str	s0, [x0]
	ldr	s0, [sp, 112]
	ldr	x0, [sp, 192]
	str	s0, [x0]
	ldr	s0, [sp, 136]
	ldr	x0, [sp, 48]
	str	s0, [x0]
	ldr	s0, [sp, 104]
	ldr	x0, [sp, 200]
	str	s0, [x0]
	ldr	s0, [sp, 128]
	ldr	x0, [sp, 40]
	str	s0, [x0]
	ldr	s0, [sp, 96]
	ldr	x0, [sp, 208]
	str	s0, [x0]
	ldr	s0, [sp, 148]
	ldr	x0, [sp, 32]
	str	s0, [x0]
	ldr	s0, [sp, 116]
	ldr	x0, [sp, 216]
	str	s0, [x0]
	ldr	s0, [sp, 140]
	ldr	x0, [sp, 24]
	str	s0, [x0]
	ldr	s0, [sp, 108]
	ldr	x0, [sp, 224]
	str	s0, [x0]
	ldr	s0, [sp, 132]
	ldr	x0, [sp, 16]
	str	s0, [x0]
	ldr	s0, [sp, 100]
	ldr	x0, [sp, 232]
	str	s0, [x0]
	ldr	s0, [sp, 124]
	ldr	x0, [sp, 176]
	str	s0, [x0]
	ldr	s0, [sp, 92]
	ldr	x0, [sp, 240]
	str	s0, [x0]
	nop
	ldp	x29, x30, [sp], 176
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE21:
	.size	scalar_fft8_soa, .-scalar_fft8_soa
	.section	.rodata
	.align	3
.LC2:
	.string	"hej"
	.align	3
.LC3:
	.string	"scalar:"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB22:
	.cfi_startproc
	sub	sp, sp, #480
	.cfi_def_cfa_offset 480
	stp	x29, x30, [sp, 80]
	.cfi_offset 29, -400
	.cfi_offset 30, -392
	add	x29, sp, 80
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	puts
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	add	x0, sp, 416
	ldp	x2, x3, [x1]
	stp	x2, x3, [x0]
	ldp	x2, x3, [x1, 16]
	stp	x2, x3, [x0, 16]
	ldp	x2, x3, [x1, 32]
	stp	x2, x3, [x0, 32]
	ldp	x2, x3, [x1, 48]
	stp	x2, x3, [x0, 48]
	add	x0, sp, 160
	mov	x1, 256
	mov	x2, x1
	mov	w1, 0
	bl	memset
	stp	xzr, xzr, [sp, 96]
	stp	xzr, xzr, [sp, 112]
	stp	xzr, xzr, [sp, 128]
	stp	xzr, xzr, [sp, 144]
	adrp	x0, .LC3
	add	x0, x0, :lo12:.LC3
	bl	puts
	add	x0, sp, 96
	add	x7, x0, 24
	add	x0, sp, 96
	add	x6, x0, 20
	add	x0, sp, 96
	add	x5, x0, 16
	add	x0, sp, 96
	add	x4, x0, 12
	add	x0, sp, 96
	add	x3, x0, 8
	add	x0, sp, 96
	add	x2, x0, 4
	add	x1, sp, 96
	add	x8, sp, 416
	add	x0, sp, 96
	add	x0, x0, 60
	str	x0, [sp, 64]
	add	x0, sp, 96
	add	x0, x0, 56
	str	x0, [sp, 56]
	add	x0, sp, 96
	add	x0, x0, 52
	str	x0, [sp, 48]
	add	x0, sp, 96
	add	x0, x0, 48
	str	x0, [sp, 40]
	add	x0, sp, 96
	add	x0, x0, 44
	str	x0, [sp, 32]
	add	x0, sp, 96
	add	x0, x0, 40
	str	x0, [sp, 24]
	add	x0, sp, 96
	add	x0, x0, 36
	str	x0, [sp, 16]
	add	x0, sp, 96
	add	x0, x0, 32
	str	x0, [sp, 8]
	add	x0, sp, 96
	add	x0, x0, 28
	str	x0, [sp]
	mov	x0, x8
	bl	scalar_fft8_soa
	add	x0, sp, 96
	mov	w1, 16
	bl	print_array_f
	mov	w0, 0
	ldp	x29, x30, [sp, 80]
	add	sp, sp, 480
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
	.text
	.section	.rodata
	.align	3
	.type	t1.0, %object
	.size	t1.0, 256
t1.0:
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
