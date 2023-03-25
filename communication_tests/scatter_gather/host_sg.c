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

typedef struct sg_xfer_context {
  size_t **metadata;          /* [in] array of block lengths */
  uint8_t ***block_addresses; /* [in] indexes to store the next block */
} sg_xfer_context;

bool get_block(struct sg_block_info *out, uint32_t dpu_index,
               uint32_t block_index, void *args) {
  if (block_index >= NB_BLOCKS) {
    return false;
  }

  /* Unpack the arguments */
  sg_xfer_context *sc_args = (sg_xfer_context *)args;
  size_t **metadata = sc_args->metadata;
  size_t length = metadata[dpu_index][block_index];
  uint8_t ***block_addresses = sc_args->block_addresses;

  /* Set the output block */
  out->length = length * sizeof(int);
  out->addr = block_addresses[dpu_index][block_index];

  return true;
}

void interface_scatter(void* arr, int num_elems, int types_size){
}
void* interface_gather(void* arr, int num_elems, int types_size){
    return NULL;
}

int main(int argc, char *argv[]){
    int dpu_number = 4;
    int nr_elements = 100;
    struct dpu_set_t set, dpu;
    
    DPU_ASSERT(dpu_alloc(num_dpus, "sgXferEnable=true", &set));

    T* A = (T*)malloc(nr_elements*sizeof(T));
    for(int i=0; i<nr_elements; i++){
        A[i] = i;
    }

    T* B = (T*)malloc(nr_elements*sizeof(T));

    //DPU_ASSERT(dpu_alloc(dpu_number, NULL, &set));
    //DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    small_table_scatter("t1", A, nr_elements, sizeof(T), 0, table_management);

    //DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    T* res = small_table_gather("t1", table_management);
    for(int i=0; i<nr_elements; i++){
        printf("%d\n", res[i]);
    }
}