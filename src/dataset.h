// Loading the dataset and saving the labels.
//
// File format (little-endian), the same for datasets and labels:
//   int32 columns, int32 rows, then rows x columns values in row-major order.
// Dataset values are float32 or float64 (detected from the file size) and are
// converted to real_t; labels are int32, one column.
#ifndef DBSCAN_DATASET_H
#define DBSCAN_DATASET_H

#include "common.h"

typedef struct {
    real_t *data; // n rows of `stride` values, SIMD_ALIGN-aligned
    int n;        // number of points
    int d;        // dimensions in the file
    int stride;   // d rounded up to a multiple of SIMD_LANES; the extra columns are 0
} Dataset;

static inline const real_t *dataset_row(const Dataset *ds, int i)
{
    return ds->data + (size_t)i * (size_t)ds->stride;
}

// They print an error and exit on failure.
void dataset_load(const char *path, Dataset *ds);
void dataset_free(Dataset *ds);
void labels_save(const char *path, const int *labels, int n);

#endif
