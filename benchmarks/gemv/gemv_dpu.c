#include <stdio.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>

#include "../../lib/Common.h"
#include "../../lib/Parallel.h"
#include "../../lib/Structs.h"
#include "../../lib/StructsPIM.h"
#include "../../lib/Table.h"
#include "Param.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__dma_aligned void* input_vec_data;
__mram_ptr void* HEAP_PTR;
uint32_t X_dim;


void to_result_vectors(void* input, void* res){
    // the data is preserved and later added to corresponding weights 
    T* res_ptr = (T*)res;
    T* matrix_ptr = (T*)input;
    T* input_vec_ptr = (T*)input_vec_data;
    *res_ptr = 0.0;


    // calculate gradients w.r.t. linear weights
    for(int i=0; i<X_dim; i++){
        *res_ptr += matrix_ptr[i] * input_vec_ptr[i];
    }

}

BARRIER_INIT(my_barrier, NR_TASKLETS);
int main() {
    int pid = me();
    if (pid == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    
    uint32_t input_vec_data_len = DPU_INPUT_ARGUMENTS.data_len;
    uint32_t input_vec_type_size = DPU_INPUT_ARGUMENTS.data_type_size;
    uint32_t input_vec_data_offset = DPU_INPUT_ARGUMENTS.data_start_offset;
    uint32_t end_offset = DPU_INPUT_ARGUMENTS.end_offset;
    uint32_t aligned_vec_size = input_vec_data_len*input_vec_type_size + (input_vec_data_len*input_vec_type_size)%8;
    
    if(pid==0){
        // initialise centroids
        fsb_allocator_t vec_allocator = fsb_alloc(aligned_vec_size, 1);
        input_vec_data = (void*)fsb_get(vec_allocator);
        X_dim = input_vec_type_size/sizeof(T);
    }

    barrier_wait(&my_barrier);

    load_arr_aligned(input_vec_data, DPU_MRAM_HEAP_POINTER+input_vec_data_offset, aligned_vec_size);
    map_dpu(DPU_MRAM_HEAP_POINTER + end_offset, to_result_vectors, &DPU_INPUT_ARGUMENTS);


    return 0;
}