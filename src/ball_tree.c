#include "ball_tree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Below this many points a subtree is built by the thread that reached it:
// smaller OpenMP tasks would cost more than they save.
#define TASK_CUTOFF 4096

typedef struct {
    const Dataset *ds;
    int *order;
    int leaf_size;
    dist2_fn dist2;
} BuildContext;

static void centroid(const BuildContext *ctx, int start, int end, real_t *center)
{
    int stride = ctx->ds->stride;
    // Accumulate in double: in single precision, large sums would lose the small addends
    double *sum = xmalloc((size_t)stride * sizeof(double));
    memset(sum, 0, (size_t)stride * sizeof(double));
    for (int k = start; k < end; k++) {
        const real_t *x = dataset_row(ctx->ds, ctx->order[k]);
        for (int j = 0; j < stride; j++)
            sum[j] += x[j];
    }
    for (int j = 0; j < stride; j++)
        center[j] = (real_t)(sum[j] / (end - start));
    free(sum);
}

// Index of the point of the range farthest from `from`; its squared distance goes in *max_d2.
static int farthest(const BuildContext *ctx, int start, int end, const real_t *from, real_t *max_d2)
{
    int best = ctx->order[start];
    real_t best_d2 = -1;
    for (int k = start; k < end; k++) {
        int q = ctx->order[k];
        real_t d2 = ctx->dist2(dataset_row(ctx->ds, q), from, ctx->ds->stride);
        if (d2 > best_d2) {
            best_d2 = d2;
            best = q;
        }
    }
    *max_d2 = best_d2;
    return best;
}

static BallNode *build_node(const BuildContext *ctx, int start, int end)
{
    const Dataset *ds = ctx->ds;
    BallNode *node = xmalloc(sizeof *node);
    node->start = start;
    node->end = end;
    node->left = node->right = NULL;
    node->center = xaligned_alloc((size_t)ds->stride * sizeof(real_t));
    centroid(ctx, start, end, node->center);

    real_t max_d2;
    int p1 = farthest(ctx, start, end, node->center, &max_d2);
    node->radius = (real_t)sqrt((double)max_d2);
    // A leaf: few points, or all of them identical and impossible to split
    if (end - start <= ctx->leaf_size || max_d2 == 0)
        return node;

    // Split around two far-apart points: p1, the farthest from the centroid, and
    // p2, the farthest from p1. Each point goes with the nearer of the two, by an
    // in-place partition of the range, so no per-node lists are allocated.
    const real_t *a = dataset_row(ds, p1);
    real_t unused;
    const real_t *b = dataset_row(ds, farthest(ctx, start, end, a, &unused));
    int i = start, j = end - 1;
    while (i <= j) {
        const real_t *x = dataset_row(ds, ctx->order[i]);
        if (ctx->dist2(x, a, ds->stride) < ctx->dist2(x, b, ds->stride)) {
            i++;
        } else {
            int tmp = ctx->order[i];
            ctx->order[i] = ctx->order[j];
            ctx->order[j--] = tmp;
        }
    }
    if (i == start || i == end) // only possible when rounding makes p1 and p2 indistinguishable
        return node;

#pragma omp task shared(node) if (i - start > TASK_CUTOFF)
    node->left = build_node(ctx, start, i);
#pragma omp task shared(node) if (end - i > TASK_CUTOFF)
    node->right = build_node(ctx, i, end);
#pragma omp taskwait
    return node;
}

void ball_tree_build(BallTree *tree, const Dataset *ds, int leaf_size, dist2_fn dist2)
{
    tree->leaf_size = leaf_size;
    tree->order = xmalloc((size_t)ds->n * sizeof(int));
    for (int i = 0; i < ds->n; i++)
        tree->order[i] = i;

    BuildContext ctx = {ds, tree->order, leaf_size, dist2};
#pragma omp parallel
#pragma omp single
    tree->root = build_node(&ctx, 0, ds->n);

    size_t row_size = (size_t)ds->stride * sizeof(real_t);
    tree->points = xaligned_alloc((size_t)ds->n * row_size);
    for (int k = 0; k < ds->n; k++)
        memcpy(tree->points + (size_t)k * ds->stride, dataset_row(ds, tree->order[k]), row_size);
}

static void free_node(BallNode *node)
{
    if (node == NULL)
        return;
    free_node(node->left);
    free_node(node->right);
    free(node->center);
    free(node);
}

void ball_tree_free(BallTree *tree)
{
    free_node(tree->root);
    free(tree->order);
    free(tree->points);
    tree->root = NULL;
    tree->order = NULL;
    tree->points = NULL;
}

static void shape_of(const BallNode *node, int depth, BallTreeShape *shape)
{
    shape->nodes++;
    if (depth > shape->depth)
        shape->depth = depth;
    if (node->left == NULL) {
        shape->leaves++;
        return;
    }
    shape_of(node->left, depth + 1, shape);
    shape_of(node->right, depth + 1, shape);
}

BallTreeShape ball_tree_shape(const BallTree *tree)
{
    BallTreeShape shape = {0, 0, 0};
    shape_of(tree->root, 0, &shape);
    return shape;
}

typedef struct {
    const BallTree *tree;
    const Dataset *ds;
    const real_t *x; // query point
    int p;           // its index, excluded from the result
    real_t eps, eps2;
    dist2_fn dist2;
    PointList *out;
    long long evals;
} Query;

static void query_node(Query *q, const BallNode *node)
{
    const int *order = q->tree->order;
    real_t to_center = (real_t)sqrt((double)q->dist2(q->x, node->center, q->ds->stride));
    q->evals++;

    // The ball is entirely outside the neighbourhood
    if (to_center > node->radius + q->eps)
        return;

    // The ball is entirely inside: take all its points without computing their distances
    if (to_center + node->radius <= q->eps) {
        for (int k = node->start; k < node->end; k++)
            if (order[k] != q->p)
                point_list_push(q->out, order[k]);
        return;
    }

    if (node->left != NULL) {
        query_node(q, node->left);
        query_node(q, node->right);
        return;
    }

    for (int k = node->start; k < node->end; k++) {
        int r = order[k];
        if (r == q->p)
            continue;
        if (q->dist2(q->x, q->tree->points + (size_t)k * q->ds->stride, q->ds->stride) <= q->eps2)
            point_list_push(q->out, r);
        q->evals++;
    }
}

long long ball_tree_query(const BallTree *tree, const Dataset *ds, int p, real_t eps, dist2_fn dist2, PointList *out)
{
    Query q = {tree, ds, dataset_row(ds, p), p, eps, eps * eps, dist2, out, 0};
    query_node(&q, tree->root);
    return q.evals;
}
