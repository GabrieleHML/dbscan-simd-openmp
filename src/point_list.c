#include "point_list.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"

void point_list_init(PointList *list, int initial_capacity)
{
    list->capacity = initial_capacity > 0 ? initial_capacity : 16;
    list->size = 0;
    list->items = xmalloc((size_t)list->capacity * sizeof(int));
}

void point_list_free(PointList *list)
{
    free(list->items);
    list->items = NULL;
    list->size = list->capacity = 0;
}

// Doubling the capacity keeps the amortized cost of a push constant.
static void reserve(PointList *list, int needed)
{
    if (needed <= list->capacity)
        return;
    int capacity = list->capacity;
    while (capacity < needed)
        capacity *= 2;
    list->items = xrealloc(list->items, (size_t)capacity * sizeof(int));
    list->capacity = capacity;
}

void point_list_push(PointList *list, int point)
{
    reserve(list, list->size + 1);
    list->items[list->size++] = point;
}

void point_list_append(PointList *list, const int *points, int count)
{
    reserve(list, list->size + count);
    memcpy(list->items + list->size, points, (size_t)count * sizeof(int));
    list->size += count;
}
