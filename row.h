#ifndef ROW_ROW_H
#define ROW_ROW_H

#include <sys/types.h>

typedef struct Mat Mat;
typedef struct Row {
    size_t cols;
    float *data;
} Row;


#define ROW_AT(r, col) (r).data[col]
Mat row_as_mat(Row row);
#define row_alloc(cols) mat_row(mat_alloc(1, cols), 0)
#define row_free(r) mat_free(row_as_mat(r))
Row row_slice(Row row, size_t i, size_t cols); // TODO
#define row_rand(r, min, max) mat_rand(row_as_mat(r), min, max)
#define row_fill(r, x) mat_fill(row_as_mat(r), x);
#define row_print(r, padding) mat_print(row_as_mat(r), padding)
#define row_copy(dst, src) mat_copy(row_as_mat(dst), row_as_mat(src))
#define row_zero(r) mat_zero(row_as_mat(r))

#endif
