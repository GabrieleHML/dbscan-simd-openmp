// Hot loops, with a hand-written SIMD version (src/asm) and a portable C one.
#ifndef DBSCAN_KERNELS_H
#define DBSCAN_KERNELS_H

#include "common.h"

// Squared Euclidean distance between two points of `d` coordinates.
// DBSCAN only compares distances, so the square root is taken only where a real
// distance is needed (ball radii and pruning).
// The SIMD version requires SIMD_ALIGN-aligned pointers and d a positive multiple
// of SIMD_LANES, which the padded dataset rows and the ball centres guarantee.
typedef real_t (*dist2_fn)(const real_t *a, const real_t *b, int d);

real_t dist2_simd(const real_t *a, const real_t *b, int d); // x86-32 SSE / x86-64 AVX
real_t dist2_c(const real_t *a, const real_t *b, int d);

// Sets labels[0..n) to `value`; the SIMD version requires SIMD_ALIGN-aligned labels.
void fill_labels_simd(int *labels, int n, int value);
void fill_labels_c(int *labels, int n, int value);

#endif
