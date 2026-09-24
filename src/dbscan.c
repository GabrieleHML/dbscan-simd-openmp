// DBSCAN in two phases, so that the expensive part runs in parallel:
//   1. the Eps-neighbourhood of every point, one independent query per point;
//   2. the expansion of the clusters from the core points, a sequential graph
//      visit that only reads the neighbourhoods computed in phase 1.
// Every point is queried exactly once, as in the original algorithm, where
// neighbourhoods were computed on demand during the expansion.
#include "dbscan.h"

#include <stdlib.h>
#include <string.h>

#include "ball_tree.h"
#include "kernels.h"
#include "point_list.h"

static long long linear_query(const Dataset *ds, int p, real_t eps2, dist2_fn dist2, PointList *out)
{
    const real_t *x = dataset_row(ds, p);
    for (int q = 0; q < ds->n; q++)
        if (q != p && dist2(x, dataset_row(ds, q), ds->stride) <= eps2)
            point_list_push(out, q);
    return ds->n - 1; // every other point
}

void dbscan(const Dataset *ds, const DbscanParams *params, int *labels, DbscanStats *stats)
{
    int n = ds->n;
    dist2_fn dist2 = params->use_simd ? dist2_simd : dist2_c;
    void (*fill_labels)(int *, int, int) = params->use_simd ? fill_labels_simd : fill_labels_c;
    real_t eps2 = params->eps * params->eps;
    memset(stats, 0, sizeof *stats);

    double t0 = wall_time();
    BallTree tree = {0};
    if (params->index == INDEX_BALL_TREE)
        ball_tree_build(&tree, ds, params->leaf_size, dist2);
    double t1 = wall_time();
    if (params->index == INDEX_BALL_TREE) {
        BallTreeShape shape = ball_tree_shape(&tree);
        stats->tree_nodes = shape.nodes;
        stats->tree_leaves = shape.leaves;
        stats->tree_depth = shape.depth;
    }

    // Phase 1. Only the neighbourhoods of core points are kept: the expansion never
    // reads the others, and dropping them saves most of the memory on sparse data.
    int **neighbours = xmalloc((size_t)n * sizeof(int *));
    int *count = xmalloc((size_t)n * sizeof(int));
    char *is_core = xmalloc((size_t)n);
    long long evals = 0, kept = 0;
    int core_points = 0;
#pragma omp parallel reduction(+ : evals, kept, core_points)
    {
        PointList found; // reused by all the queries of a thread
        point_list_init(&found, 256);
#pragma omp for schedule(dynamic, 64)
        for (int p = 0; p < n; p++) {
            point_list_clear(&found);
            evals += params->index == INDEX_BALL_TREE ? ball_tree_query(&tree, ds, p, params->eps, dist2, &found)
                                                      : linear_query(ds, p, eps2, dist2, &found);
            is_core[p] = found.size + 1 >= params->min_pts;
            count[p] = is_core[p] ? found.size : 0;
            neighbours[p] = NULL;
            if (is_core[p] && found.size > 0) {
                neighbours[p] = xmalloc((size_t)found.size * sizeof(int));
                memcpy(neighbours[p], found.items, (size_t)found.size * sizeof(int));
                kept += found.size;
            }
            core_points += is_core[p];
        }
        point_list_free(&found);
    }
    double t2 = wall_time();

    // Phase 2. A cluster starts from the first core point not yet labelled and
    // grows through the neighbourhoods of its core points (depth-first, with a stack).
    fill_labels(labels, n, UNCLASSIFIED);
    PointList stack;
    point_list_init(&stack, 1024);
    int cluster = 0;
    for (int p = 0; p < n; p++) {
        if (labels[p] != UNCLASSIFIED || !is_core[p])
            continue;
        labels[p] = cluster;
        point_list_push(&stack, p);
        while (stack.size > 0) {
            int q = point_list_pop(&stack); // always a core point
            for (int k = 0; k < count[q]; k++) {
                int r = neighbours[q][k];
                if (labels[r] != UNCLASSIFIED)
                    continue;
                labels[r] = cluster;
                if (is_core[r])
                    point_list_push(&stack, r);
            }
        }
        cluster++;
    }
    int noise_points = 0;
    for (int p = 0; p < n; p++) {
        if (labels[p] == UNCLASSIFIED) {
            labels[p] = NOISE;
            noise_points++;
        }
    }
    double t3 = wall_time();

    point_list_free(&stack);
    for (int p = 0; p < n; p++)
        free(neighbours[p]);
    free(neighbours);
    free(count);
    free(is_core);
    if (params->index == INDEX_BALL_TREE)
        ball_tree_free(&tree);

    stats->clusters = cluster;
    stats->core_points = core_points;
    stats->noise_points = noise_points;
    stats->distance_evals = evals;
    stats->neighbours = kept;
    stats->index_time = t1 - t0;
    stats->neighbour_time = t2 - t1;
    stats->expand_time = t3 - t2;
    stats->total_time = t3 - t0;
}
