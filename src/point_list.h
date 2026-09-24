// Growable array of point indices, used for neighbourhoods and as a stack.
#ifndef DBSCAN_POINT_LIST_H
#define DBSCAN_POINT_LIST_H

typedef struct {
    int *items;
    int size;
    int capacity;
} PointList;

void point_list_init(PointList *list, int initial_capacity);
void point_list_free(PointList *list);
void point_list_push(PointList *list, int point);
void point_list_append(PointList *list, const int *points, int count);

static inline int point_list_pop(PointList *list)
{
    return list->items[--list->size]; // caller checks size > 0
}

static inline void point_list_clear(PointList *list)
{
    list->size = 0;
}

#endif
