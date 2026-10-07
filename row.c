#include "row.h"
#include "mat.h"

Mat row_as_mat(Row row)
{
    return (Mat) {
        .rows = 1,
        .cols = row.cols,
        .data = row.data,
    };
}
