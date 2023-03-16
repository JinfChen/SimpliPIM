#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <dpu.h>

#include "../../lib/UpmemCustom.h"
#include "../../lib/Structs.h"
#include "../../lib/Common.h"
#include "../../lib/timer.h"
#include "Param.h"

#define DPU_BINARY "bin/dpu_binary"


void save_data(T* input, T* querys){
    FILE* fp = fopen ("bin/data.csv", "w");
    fprintf(fp,"%llu %llu\n", nr_elements, num_querys);
    for(int i=0; i<nr_elements; i++){
        fprintf(fp,"%lld\n", input[i]);
    }

    for (uint64_t i = 0; i < num_querys; i++) {
		fprintf(fp,"%lld\n", querys[i]);
	}
    fclose(fp);
}

void init_data(T* input, T* querys){
    input[0] = 1;
	for (uint64_t i = 1; i < nr_elements; i++) {
		input[i] = input[i - 1] + (rand() % 10) + 1;
	}
	for (uint64_t i = 0; i < num_querys; i++) {
		querys[i] = input[rand() % (nr_elements - 2)];
	}
}

void add(void* p1, void* p2){
    T* ptr1 = (T*)p1;
    T* ptr2 = (T*)p2;
    *ptr1 += *ptr2;
}

uint64_t binarySearch(T * input, T* querys)
{

	uint64_t found = -1;
    uint64_t right = nr_elements-1;
	uint64_t q, r, l, m;
	
       #pragma omp parallel for private(q,r,l,m)
     	for(q = 0; q < num_querys; q++)
      	{
		l = 0;
		r = right;
		while (l <= r) 
		{
	    		m = l + (r - l) / 2;

	    		// Check if x is present at mid
	     		if (input[m] == querys[q])
			{	
		    		found += m;
				break;
			}
	    		// If x greater, ignore left half
	    		if (input[m] < querys[q])
			    	l = m + 1;

	    		// If x is smaller, ignore right half
			else
		    		r = m - 1;
		
		}
       	}

      	return found;
}

void run(){

	struct dpu_set_t set, dpu;
    uint32_t num_dpus;

    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &set));
    //DPU_ASSERT(dpu_alloc(3, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(set, &num_dpus));


    T* inputs = (T*)malloc_broadcast_aligned(nr_elements, sizeof(T));
    T* queries = (T*)malloc_split_aligned( num_querys, sizeof(T), num_dpus);
    init_data(inputs, queries);
    uint64_t correct_res = binarySearch(inputs, queries);
    printf("the correct result is %d\n", correct_res);
    save_data(inputs, queries);

	Timer timer;
    start(&timer, 0, 0);
    start(&timer, 5, 0);
    uint32_t data_offset = host_split_to_dpu(set, queries, num_querys, sizeof(T), num_dpus, 0);
	uint32_t end_offset =  host_broadcast_to_dpu(set, inputs, nr_elements, sizeof(T), data_offset);
	
	dpu_arguments_t* input_args = (dpu_arguments_t*) malloc(num_dpus * sizeof(dpu_arguments_t));
  	for(int i=0; i<num_dpus; i++){
     	input_args[i].input_start_offset = 0;
     	input_args[i].input_type_size = sizeof(T);
     	input_args[i].data_start_offset = data_offset;
     	input_args[i].data_len = (uint32_t)nr_elements;
     	input_args[i].data_type_size = sizeof(T);
     	input_args[i].end_offset = end_offset;
     	input_args[i].table_type_size = sizeof(T);
     	input_args[i].table_len = 1;
  	}
	prepare_input_len_and_parse_args(set, input_args, (uint32_t)num_querys, sizeof(T), num_dpus);
	stop(&timer, 0);

	start(&timer, 1, 0);
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    stop(&timer, 1);

    if(print_info){
      DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
      }
    }

	start(&timer, 2, 0);
    T* res = (T*)malloc(sizeof(T));
    gather_tables_to_host(set, res, 1, sizeof(T), end_offset, num_dpus, zero_init, add);
	*res -= 1;
    stop(&timer, 2);
    stop(&timer, 5);

	if(*res == correct_res){
		printf("got correct results\n");
	}
	else{
		printf("expected %d, got %d\n", correct_res, *res);
	}

	//print timing
	printf("initial CPU-DPU input transfer (ms): ");
	print(&timer, 0, 1);
  	printf("\n");
	printf("DPU Kernel Time (ms): ");
	print(&timer, 1, 1);
  	printf("\n");
	printf("DPU-CPU Time (ms): ");
	print(&timer, 2, 1);
  	printf("\n");
	
	float total_time = timer.time[0];
  	for(int i=1; i<3; i++){
    	total_time += timer.time[i];
  	}
  	printf("total time added up (ms): %f\n", total_time/1000);
}

int main(int argc, char *argv[]){
	srand(17); 
  	run();
  	return 0;
}