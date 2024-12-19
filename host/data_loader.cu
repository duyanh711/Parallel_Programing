#include "data_loader.cuh"
#include <cstdio>

void DataLoader::load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Không thể mở file\n");
        return;
    }
    
    size_t elements_read = fread(data, sizeof(float), size, file);
    if (elements_read != size) {
        printf("Số phần tử đọc được không khớp\n");
    }
    
    fclose(file);
}

void DataLoader::load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Không thể mở file\n");
        return;
    }
    
    size_t elements_read = fread(labels, sizeof(int), size, file);
    if (elements_read != size) {
        printf("Số phần tử đọc được không khớp\n");
    }
    
    fclose(file);
} 