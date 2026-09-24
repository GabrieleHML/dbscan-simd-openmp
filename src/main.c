// Command-line interface: loads a dataset, runs DBSCAN and reports labels and timings.
#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dataset.h"
#include "dbscan.h"

#if defined(__x86_64__)
#define BUILD "x86-64 AVX"
#else
#define BUILD "x86-32 SSE"
#endif

#define DEFAULT_LEAF_SIZE 32

static void usage(FILE *f, const char *prog)
{
    fprintf(f,
            "Usage: %s -i FILE -e EPS -m MIN_PTS [options]\n"
            "\n"
            "Clusters the points of FILE with DBSCAN (" BUILD " build).\n"
            "\n"
            "  -i, --input FILE     dataset: int32 columns, int32 rows, float32/float64 values\n"
            "  -e, --eps EPS        neighbourhood radius\n"
            "  -m, --min-pts N      points in a neighbourhood, itself included, to be a core point\n"
            "  -o, --output FILE    write the labels (-1 = noise, clusters from 0)\n"
            "      --index KIND     neighbourhood search: balltree (default) or linear\n"
            "      --leaf-size N    maximum points in a ball tree leaf (default %d)\n"
            "      --no-simd        use the C kernels instead of the SSE/AVX ones\n"
            "  -t, --threads N      OpenMP threads (default: all the cores)\n"
            "      --json           print the results as one line of JSON\n"
            "  -q, --quiet          print only errors (and the JSON line, if requested)\n"
            "  -h, --help           show this help\n",
            prog, DEFAULT_LEAF_SIZE);
}

_Noreturn static void bad_usage(const char *prog, const char *message, const char *arg)
{
    fprintf(stderr, "%s: %s%s%s\n", prog, message, arg ? ": " : "", arg ? arg : "");
    fprintf(stderr, "Try '%s --help' for more information.\n", prog);
    exit(2);
}

static double parse_double(const char *prog, const char *s, const char *what)
{
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !(v > 0))
        bad_usage(prog, what, s);
    return v;
}

static int parse_int(const char *prog, const char *s, const char *what)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0 || v > 1 << 30)
        bad_usage(prog, what, s);
    return (int)v;
}

int main(int argc, char **argv)
{
    const char *prog = argv[0];
    const char *input = NULL, *output = NULL;
    DbscanParams params = {0, 0, INDEX_BALL_TREE, DEFAULT_LEAF_SIZE, 1};
    int threads = 0, json = 0, quiet = 0;

    enum { OPT_INDEX = 256, OPT_LEAF_SIZE, OPT_NO_SIMD, OPT_JSON };
    static const struct option options[] = {
        {"input", required_argument, NULL, 'i'},
        {"eps", required_argument, NULL, 'e'},
        {"min-pts", required_argument, NULL, 'm'},
        {"output", required_argument, NULL, 'o'},
        {"index", required_argument, NULL, OPT_INDEX},
        {"leaf-size", required_argument, NULL, OPT_LEAF_SIZE},
        {"no-simd", no_argument, NULL, OPT_NO_SIMD},
        {"threads", required_argument, NULL, 't'},
        {"json", no_argument, NULL, OPT_JSON},
        {"quiet", no_argument, NULL, 'q'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "i:e:m:o:t:qh", options, NULL)) != -1) {
        switch (opt) {
        case 'i': input = optarg; break;
        case 'e': params.eps = (real_t)parse_double(prog, optarg, "EPS must be a positive number"); break;
        case 'm': params.min_pts = parse_int(prog, optarg, "MIN_PTS must be a positive integer"); break;
        case 'o': output = optarg; break;
        case 't': threads = parse_int(prog, optarg, "--threads must be a positive integer"); break;
        case 'q': quiet = 1; break;
        case 'h': usage(stdout, prog); return 0;
        case OPT_LEAF_SIZE: params.leaf_size = parse_int(prog, optarg, "--leaf-size must be a positive integer"); break;
        case OPT_NO_SIMD: params.use_simd = 0; break;
        case OPT_JSON: json = 1; break;
        case OPT_INDEX:
            if (strcmp(optarg, "balltree") == 0)
                params.index = INDEX_BALL_TREE;
            else if (strcmp(optarg, "linear") == 0)
                params.index = INDEX_LINEAR;
            else
                bad_usage(prog, "the index must be 'balltree' or 'linear'", optarg);
            break;
        default: bad_usage(prog, "invalid option", NULL);
        }
    }
    if (optind < argc)
        bad_usage(prog, "unexpected argument", argv[optind]);
    if (input == NULL || params.eps == 0 || params.min_pts == 0)
        bad_usage(prog, "the options -i, -e and -m are required", NULL);

#ifdef _OPENMP
    if (threads > 0)
        omp_set_num_threads(threads);
    threads = omp_get_max_threads();
#else
    if (threads > 1)
        fprintf(stderr, "%s: warning: built without OpenMP, running on 1 thread\n", prog);
    threads = 1;
#endif

    Dataset ds;
    dataset_load(input, &ds);
    int *labels = xaligned_alloc((size_t)ds.n * sizeof(int));
    DbscanStats st;
    dbscan(&ds, &params, labels, &st);
    if (output != NULL)
        labels_save(output, labels, ds.n);

    const char *index = params.index == INDEX_BALL_TREE ? "balltree" : "linear";
    if (!quiet) {
        printf("dataset     %s: %d points, %d dimensions\n", input, ds.n, ds.d);
        printf("parameters  eps=%g min_pts=%d index=%s", (double)params.eps, params.min_pts, index);
        if (params.index == INDEX_BALL_TREE)
            printf(" leaf_size=%d", params.leaf_size);
        printf(" simd=%s threads=%d build=%s\n", params.use_simd ? "on" : "off", threads, BUILD);
        printf("clusters    %d (%d core points, %d noise points)\n", st.clusters, st.core_points, st.noise_points);
        printf("time        %.4f s (index %.4f, neighbourhoods %.4f, expansion %.4f)\n", st.total_time, st.index_time,
               st.neighbour_time, st.expand_time);
        printf("distances   %lld computed\n", st.distance_evals);
        if (params.index == INDEX_BALL_TREE)
            printf("ball tree   %d nodes, %d leaves, depth %d\n", st.tree_nodes, st.tree_leaves, st.tree_depth);
        if (output != NULL)
            printf("labels      written to %s\n", output);
    }
    if (json) {
        printf("{\"build\":\"%s\",\"n\":%d,\"d\":%d,\"eps\":%g,\"min_pts\":%d,\"index\":\"%s\",\"leaf_size\":%d,"
               "\"simd\":%s,\"threads\":%d,\"clusters\":%d,\"core_points\":%d,\"noise_points\":%d,"
               "\"distance_evals\":%lld,\"neighbours\":%lld,\"tree_nodes\":%d,\"tree_depth\":%d,"
               "\"index_time\":%.6f,\"neighbour_time\":%.6f,\"expand_time\":%.6f,\"total_time\":%.6f}\n",
               BUILD, ds.n, ds.d, (double)params.eps, params.min_pts, index, params.leaf_size,
               params.use_simd ? "true" : "false", threads, st.clusters, st.core_points, st.noise_points,
               st.distance_evals, st.neighbours, st.tree_nodes, st.tree_depth, st.index_time, st.neighbour_time,
               st.expand_time, st.total_time);
    }

    free(labels);
    dataset_free(&ds);
    return 0;
}
