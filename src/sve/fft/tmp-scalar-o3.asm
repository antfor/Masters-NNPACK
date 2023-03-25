	.arch armv8-a+sve
	.file	"tmp.c"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC1:
	.string	"hej"
	.align	3
.LC2:
	.string	"scalar:"
	.align	3
.LC3:
	.string	"%f "
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 4,,11
	.global	main
	.type	main, %function
main:
.LFB33:
	.cfi_startproc
	stp	x29, x30, [sp, -112]!
	.cfi_def_cfa_offset 112
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	adrp	x20, .LC3
	add	x19, sp, 48
	str	x21, [sp, 32]
	.cfi_offset 21, -80
	bl	puts
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	puts
	add	x20, x20, :lo12:.LC3
	adrp	x0, .LC4
	mov	x5, 3229614080
	mov	x3, 1120403456
	mov	x1, 3229614080
	ldr	q0, [x0, #:lo12:.LC4]
	mov	x0, 3238002688
	movk	x5, 0xf61a, lsl 32
	mov	x4, 5441723564032
	movk	x3, 0x4f3, lsl 32
	mov	x2, 270591529582592
	movk	x1, 0x4f3, lsl 32
	movk	x0, 0x827a, lsl 32
	add	x21, sp, 112
	movk	x5, 0xc015, lsl 48
	movk	x4, 0x40b5, lsl 48
	movk	x3, 0x40b5, lsl 48
	movk	x2, 0xc015, lsl 48
	movk	x1, 0xc0b5, lsl 48
	movk	x0, 0xc15a, lsl 48
	stp	x5, x4, [sp, 64]
	stp	x3, x2, [sp, 80]
	stp	x1, x0, [sp, 96]
	str	q0, [sp, 48]
	.p2align 3,,7
.L2:
	ldr	s0, [x19], 4
	mov	x0, x20
	fcvt	d0, s0
	bl	printf
	cmp	x19, x21
	bne	.L2
	mov	w0, 10
	bl	putchar
	mov	w0, 0
	ldp	x19, x20, [sp, 16]
	ldr	x21, [sp, 32]
	ldp	x29, x30, [sp], 112
	.cfi_restore 30
	.cfi_restore 29
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
.LC4:
	.word	1108344832
	.word	-1051032966
	.word	-1056964608
	.word	-1061878541
	.ident	"GCC: (GNU Toolchain for the A-profile Architecture 10.2-2020.11 (arm-10.16)) 10.2.1 20201103"
	.section	.note.GNU-stack,"",@progbits
