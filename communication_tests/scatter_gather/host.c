#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <dpu.h>

#include "../../lib/processing/ProcessingHelperHost.h"
#include "../../lib/communication/CommOps.h"
#include "../../lib/management/Management.h"

#define DPU_BINARY "bin/dpu_binary"

typedef int T; 

void interface_scatter(){
    return NULL;
}
void* interface_gather(){
    return NULL;
}

int main(int argc, char *argv[]){
    int dpu_number = 4;
    int nr_elements = 100;
    struct dpu_set_t set, dpu;
    
    smalltable_management_t* table_management = table_management_init(dpu_number);
    T* A = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);
    for(int i=0; i<nr_elements; i++){
        A[i] = i;
    }


    //DPU_ASSERT(dpu_alloc(dpu_number, NULL, &set));
    //DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    small_table_scatter("t1", A, nr_elements, sizeof(T), 0, table_management);

    //DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    T* res = small_table_gather("t1", table_management);
    for(int i=0; i<nr_elements; i++){
        printf("%d\n", res[i]);
    }
}