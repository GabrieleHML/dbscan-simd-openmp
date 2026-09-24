// DBSCAN clustering (Ester et al., KDD 1996).
#ifndef DBSCAN_DBSCAN_H
#define DBSCAN_DBSCAN_H

#include "dataset.h"

typedef enum { INDEX_LINEAR, INDEX_BALL_TREE } IndexKind;

typedef struct {
    real_t eps;      // neighbourhood radius
    int min_pts;     // neighbourhood size, the point itself included, that makes a core point
    IndexKind index; // how neighbourhoods are searched
    int leaf_size;   // ball tree only
    int use_simd;    // SSE/AVX kernels instead of the C ones
} DbscanParams;

typedef struct {
    int clusters, core_points, noise_points;
    int tree_nodes, tree_leaves, tree_depth; // ball tree only
    long long distance_evals; // distances computed by the neighbourhood queries
    long long neighbours;     // sum of the neighbourhood sizes of the core points (memory used)
    double index_time;        // building the ball tree
    double neighbour_time;    // phase 1: neighbourhood queries
    double expand_time;       // phase 2: cluster expansion
    double total_time;
} DbscanStats;

// Writes in labels[0..n) the cluster of each point (from 0) or NOISE.
// labels must be SIMD_ALIGN-aligned. Border points reachable from more than one
// cluster go to the first one found, the same rule scikit-learn applies.
void dbscan(const Dataset *ds, const DbscanParams *params, int *labels, DbscanStats *stats);

#endif
