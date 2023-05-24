
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

//--riscv-----------------------------------------------------------

static inline void rvprint_f(__epi_2xf32 printme, const int n, long gvl)
{

    float tmp[n];

    __builtin_epi_vstore_2xf32(tmp, printme, gvl);

    for (int i = 0; i < n; i++)
        printf("%f ", tmp[i]);

    printf("\n");
}

static inline void rvprint_i(__epi_2xi32 printme, const int n, long gvl)
{

    int tmp[n];

    __builtin_epi_vstore_2xi32(tmp, printme, gvl);

    for (int i = 0; i < n; i++)
        printf("%i ", tmp[i]);

    printf("\n");
}

static inline void rvprint_b(__epi_2xi1 printme, const int n, long gvl)
{

    int tmp[n];

    __builtin_epi_vstore_2xi32(tmp, __builtin_epi_cast_2xi32_2xi1(printme), gvl);

    for (int i = 0; i < n; i++)
        printf("%d ", tmp[i]);

    printf("\n");
}

//--arr-----------------------------------------------------------

static inline void print_array_cf(const float *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%f ", arr[i]);

    printf("\n");
}

static inline void print_array_f(float *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%f ", arr[i]);

    printf("\n");
}

static inline void fprint_array_f(float *arr, int n, int m)
{
    for (int i = 0; i < n/m; i++){
        for (int j = 0; j < m; j++)
            printf("%f ", arr[i *m + j]);

        printf("\n");
    }
        
       

   
}

static inline void print_transform(float transform[restrict static 1], size_t transform_stride, uint32_t block_size, uint32_t simd_width)
{
    for (size_t row = 0; row < block_size; row += 2) {
		for (size_t column = 0; column < block_size / simd_width; column += 1) {
			print_array_f(transform, simd_width);
            print_array_f(transform + simd_width, simd_width);
			transform += transform_stride;
		}
        printf("\n");
	}
}

static inline void print_array_i(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("\n");
}