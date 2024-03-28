#include "print.cuh"
#include <stdio.h>

EXPORT int print_tensor(tensor* data){
    int len = 1;

    for(int dim = 0; dim < data->dims; dim++){
        len *= data->size[dim];
    }

    if (data->on_device) {
        return ERROR_NOT_ON_HOST;
    }

    printf("Data size:");
    for (int i = 0; i < data->dims; i++){
        printf("%d ", data->size[i]);
    }
    printf("\n");


    printf("Data on host:\n");
    for (int i = 0; i < len; i++){
        printf("%f ", data->data_host[i]);
    }
    printf("\n");

    return 0;
}