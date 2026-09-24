// Portable versions of the kernels, used with --no-simd and as a reference in the tests.
#include "kernels.h"

real_t dist2_c(const real_t *a, const real_t *b, int d)
{
    real_t sum = 0;
    for (int i = 0; i < d; i++) {
        real_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void fill_labels_c(int *labels, int n, int value)
{
    for (int i = 0; i < n; i++)
        labels[i] = value;
}
