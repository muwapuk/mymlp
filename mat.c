#include "mat.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Mat mat_alloc(size_t rows, size_t cols) {
    assert(rows != 0 && cols != 0);
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.data = calloc(rows*cols, sizeof(*m.data));
    assert(m.data != NULL);

    return m;
}
Row mat_row(Mat m, size_t row)
{
    return (Row) { 
        .cols = m.cols,
        .data = &MAT_AT(m, row, 0)
    };
}
void mat_free(Mat mat)
{
    if(mat.data != NULL) {
        free(mat.data);
    }
}
static inline float rand_float()
{
    return (float) rand() / (float) RAND_MAX;
}
void mat_rand(Mat m, float min, float max)
{
    for(size_t row = 0; row < m.rows; row++) {
        for(size_t col = 0; col < m.cols; col++) {
            MAT_AT(m, row, col) = rand_float()*(max - min) + min;
        }
    }
}
void mat_dot(Mat dst, Mat a, Mat b)
{
    if(a.cols !=  b.rows 
    || dst.rows !=  a.rows
    || dst.cols !=  b.cols) {
        fprintf(stderr, "Failed to get dot prodact of matrices. Bad dimensions!\n");
        return;
    }
    size_t additions = a.cols;
    for(size_t row = 0; row < dst.rows; row++) {
        for(size_t col = 0; col < dst.cols; col++) {
            MAT_AT(dst, row, col) = 0;
            for(size_t add_num = 0; add_num < additions; add_num++) {
                MAT_AT(dst, row, col) += MAT_AT(a, row, add_num)*MAT_AT(b, add_num, col);
            }
        }
    }
}
void mat_sum(Mat dst, Mat a)
{
    if(dst.rows != a.rows || dst.cols != a.cols) {
        fprintf(stderr, "Failed to sum matrices. Rows and colums are not the same!\n");
        return;
    }
    for(size_t row = 0; row < dst.rows; row++) { 
        for(size_t col = 0; col < dst.cols; col++) {
            MAT_AT(dst, row, col) += MAT_AT(a, row, col);
        }
    }
}
void mat_print(Mat m, size_t padding)
{
    printf("%*s = [\n", (int) padding, "");
    for (size_t row = 0; row < m.rows; ++row) {
        printf("%*s    ", (int) padding, "");
        for (size_t col = 0; col < m.cols; ++col) {
            printf("%f ", MAT_AT(m, row, col));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}
void mat_zero(Mat m)
{
    memset(m.data, 0, sizeof(*m.data)*m.rows*m.cols);
}
