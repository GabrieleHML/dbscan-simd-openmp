// Ball tree: a binary tree of hyperspheres ("balls") that answers the
// Eps-neighbourhood queries of DBSCAN while skipping most of the dataset.
#ifndef DBSCAN_BALL_TREE_H
#define DBSCAN_BALL_TREE_H

#include "dataset.h"
#include "kernels.h"
#include "point_list.h"

typedef struct BallNode {
    real_t *center;                // centroid of the points, SIMD_ALIGN-aligned
    real_t radius;                 // distance from the centroid to the farthest point
    int start, end;                // the node holds order[start..end)
    struct BallNode *left, *right; // both NULL in a leaf
} BallNode;

typedef struct {
    BallNode *root;
    int *order;     // point indices, arranged so that every node owns a contiguous range
    real_t *points; // copy of the rows in the same order, so leaves are scanned sequentially
    int leaf_size;
} BallTree;

typedef struct {
    int nodes, leaves, depth;
} BallTreeShape;

void ball_tree_build(BallTree *tree, const Dataset *ds, int leaf_size, dist2_fn dist2);
void ball_tree_free(BallTree *tree);
BallTreeShape ball_tree_shape(const BallTree *tree);

// Appends to `out` every point within `eps` of point p, p excluded.
// Returns the number of distances computed, a measure of how much the tree prunes.
long long ball_tree_query(const BallTree *tree, const Dataset *ds, int p, real_t eps, dist2_fn dist2, PointList *out);

#endif
