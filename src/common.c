#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void out_of_memory(size_t size)
{
    fprintf(stderr, "error: cannot allocate %zu bytes\n", size);
    exit(EXIT_FAILURE);
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL && size > 0)
        out_of_memory(size);
    return p;
}

void *xrealloc(void *ptr, size_t size)
{
    void *p = realloc(ptr, size);
    if (p == NULL && size > 0)
        out_of_memory(size);
    return p;
}

void *xaligned_alloc(size_t size)
{
    void *p = NULL;
    if (posix_memalign(&p, SIMD_ALIGN, size > 0 ? size : SIMD_ALIGN) != 0)
        out_of_memory(size);
    return p;
}

double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
