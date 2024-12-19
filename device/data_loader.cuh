#pragma once

#include <stdio.h>

class DataLoader {
public:
    static void load_data(const char *filename, float *data, int size);
    static void load_labels(const char *filename, int *labels, int size);
}; 