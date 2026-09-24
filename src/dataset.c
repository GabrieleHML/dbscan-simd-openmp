#include "dataset.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Noreturn static void fail(const char *path, const char *reason)
{
    fprintf(stderr, "error: %s: %s\n", path, reason);
    exit(EXIT_FAILURE);
}

void dataset_load(const char *path, Dataset *ds)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        fail(path, "cannot open file");

    int32_t header[2]; // columns, rows
    if (fread(header, sizeof(int32_t), 2, f) != 2)
        fail(path, "missing header");
    int d = header[0], n = header[1];
    if (n <= 0 || d <= 0)
        fail(path, "invalid number of rows or columns");

    // The element size is deduced from the file size, so the same file works with both builds
    fseek(f, 0, SEEK_END);
    long payload = ftell(f) - (long)sizeof(header);
    fseek(f, (long)sizeof(header), SEEK_SET);
    size_t count = (size_t)n * (size_t)d;
    size_t elem;
    if ((size_t)payload == count * sizeof(float))
        elem = sizeof(float);
    else if ((size_t)payload == count * sizeof(double))
        elem = sizeof(double);
    else
        fail(path, "size does not match the header (expected float32 or float64 values)");

    ds->n = n;
    ds->d = d;
    ds->stride = (d + SIMD_LANES - 1) / SIMD_LANES * SIMD_LANES;
    // Zero padding does not change the distances, and keeps every row aligned
    ds->data = xaligned_alloc((size_t)n * (size_t)ds->stride * sizeof(real_t));
    memset(ds->data, 0, (size_t)n * (size_t)ds->stride * sizeof(real_t));

    unsigned char *row = xmalloc((size_t)d * elem);
    for (int i = 0; i < n; i++) {
        if (fread(row, elem, (size_t)d, f) != (size_t)d)
            fail(path, "truncated data");
        real_t *dst = ds->data + (size_t)i * (size_t)ds->stride;
        for (int j = 0; j < d; j++)
            dst[j] = elem == sizeof(float) ? (real_t)((float *)row)[j] : (real_t)((double *)row)[j];
    }
    free(row);
    fclose(f);
}

void dataset_free(Dataset *ds)
{
    free(ds->data);
    ds->data = NULL;
}

void labels_save(const char *path, const int *labels, int n)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL)
        fail(path, "cannot create file");
    int32_t header[2] = {1, n};
    if (fwrite(header, sizeof(int32_t), 2, f) != 2 || fwrite(labels, sizeof(int32_t), (size_t)n, f) != (size_t)n)
        fail(path, "write error");
    fclose(f);
}
