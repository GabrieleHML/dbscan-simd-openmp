// Types and helpers shared by every module.
#ifndef DBSCAN_COMMON_H
#define DBSCAN_COMMON_H

#include <stddef.h>

// The precision follows the target, as required by the course: single precision
// with the SSE kernels on x86-32, double precision with the AVX kernels on x86-64.
#if defined(__x86_64__)
typedef double real_t;
#define SIMD_ALIGN 32 // one YMM register
#elif defined(__i386__)
typedef float real_t;
#define SIMD_ALIGN 16 // one XMM register
#else
#error "Only x86-32 and x86-64 are supported"
#endif

// Values held by one SIMD register: 4 floats in XMM, 4 doubles in YMM.
// Rows are padded to a multiple of it, so the kernels need no scalar tail.
#define SIMD_LANES 4

// Cluster ids start from 0; negative values are special, as in scikit-learn.
#define NOISE (-1)
#define UNCLASSIFIED (-2)

// Allocation helpers that print an error and exit instead of returning NULL.
void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t size);
void *xaligned_alloc(size_t size); // SIMD_ALIGN-aligned, release with free()

// Monotonic wall-clock time in seconds (clock() would add up the CPU time of all threads).
double wall_time(void);

#endif
