#ifndef MAT_MAT_H
#define MAT_MAT_H

#include <sys/types.h>

#include "row.h"

typedef struct Mat {
    size_t rows;
    size_t cols;
    float *data;
} Mat;

#define MAT_AT(m, row, col) (m).data[(row)*(m).cols + (col)]

Mat mat_alloc(size_t rows, size_t cols);
Row mat_row(Mat m, size_t row);
void mat_free(Mat m);
void mat_rand(Mat m, float min, float max);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m, size_t padding);
void mat_zero(Mat m);

#endif
