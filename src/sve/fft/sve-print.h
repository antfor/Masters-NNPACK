#pragma once

#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>


// todo remove


static inline void svprint_f(svbool_t pg, svfloat32_t printme, const int n)
{

    float32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%f ", tmp[i]);

    printf("\n");
}

static inline void svprintf_f32(svbool_t pg, svfloat32_t printme, const int n, const int m)
{

    float32_t tmp[n];

    svst1(pg, tmp, printme);
    printf("svfloat start");
    for (int i = 0; i < n; i += m)
    {
        printf("\n    ");
        for (int j = 0; j < m; j++)
            printf("%f ", tmp[i + j]);
    }
    printf("\n svfloat end \n");
}

static inline void svprint_i(svbool_t pg, svint32_t printme, const int n)
{

    int32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%d ", tmp[i]);

    printf("\n");
}

static inline void svprint_ui(svbool_t pg, svuint32_t printme, const int n)
{

    int32_t tmp[n];

    svst1(pg, tmp, printme);

    for (int i = 0; i < n; i++)
        printf("%d ", tmp[i]);

    printf("\n");
}

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