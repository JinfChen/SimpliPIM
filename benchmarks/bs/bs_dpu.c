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
__dma_aligned T* arr_data;
__mram_ptr void* HEAP_PTR;


void add(void* p1, void* p2){
    T* ptr1 = (T*)p1;
    T* ptr2 = (T*)p2;
    *ptr1 += *ptr2;
}

void binarySearch(void* input_point, void* intermediate_input, uint32_t* key)
{
    *key = 0;
    T* query = (T*)input_point;
    uint64_t* found = (uint64_t*) intermediate_input;

	*found = 0;
    uint64_t right = nr_elements-1;
	uint64_t q, r, l, m;
	
	l = 0;
	r = right;
	while (l <= r) 
		{
	    		m = l + (r - l) / 2;

	    		// Check if x is present at mid
	     		if (arr_data[m] == (*query))
			{	
		    		*found += m;
				break;
			}
	    		// If x greater, ignore left half
	    		if (arr_data[m] < (*query))
			    	l = m + 1;

	    		// If x is smaller, ignore right half
			else
		    		r = m - 1;
		
		}
    
    

}


BARRIER_INIT(my_barrier, NR_TASKLETS);
int main() {
    int pid = me();
    if (pid == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    
    // Barrier
    barrier_wait(&my_barrier);

    uint32_t arr_data_len = DPU_INPUT_ARGUMENTS.data_len;
    uint32_t arr_type_size = DPU_INPUT_ARGUMENTS.data_type_size;
    uint32_t arr_data_offset = DPU_INPUT_ARGUMENTS.data_start_offset;
    uint32_t end_offset = DPU_INPUT_ARGUMENTS.end_offset;
    HEAP_PTR = DPU_MRAM_HEAP_POINTER + end_offset;
    uint32_t aligned_arr_size = arr_data_len*arr_type_size + (arr_data_len*arr_type_size)%8;
    if(pid==0){
        // initialise arr
        fsb_allocator_t arr_allocator = fsb_alloc(aligned_arr_size, 1);
        arr_data = fsb_get(arr_allocator);
    }
    barrier_wait(&my_barrier);
    load_arr_aligned(arr_data, DPU_MRAM_HEAP_POINTER+arr_data_offset, aligned_arr_size);
    barrier_wait(&my_barrier);

    map_and_combine_oncache(DPU_MRAM_HEAP_POINTER + end_offset, zero_init, binarySearch, add, &DPU_INPUT_ARGUMENTS);
    //map_and_combine_oncache_imbalanced(DPU_MRAM_HEAP_POINTER + end_offset, zero_init, binarySearch, add, &DPU_INPUT_ARGUMENTS);
    return 0;
}