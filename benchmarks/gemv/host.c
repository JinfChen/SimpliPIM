#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <dpu.h>

#include "../../lib/UpmemCustom.h"
#include "../../lib/Structs.h"
#include "../../lib/Common.h"
#include "../../lib/timer.h"
#include "Param.h"

#define DPU_BINARY "bin/dpu_binary"

void make_matrix(T* A){
    for(unsigned long i=0; i<rows; i++){
        for(unsigned long j=0; j<cols; j++){
            A[i*cols+j] = 1.0/( (T) i + (T) j + 1.0);
        }
    }
}

void make_input_vector(T* x){
    for(int i=0; i<cols; i++){
        x[i] = (T)(i+1);
    }
}

void make_zero_vector(T* b, uint32_t len){
    for(int i=0; i<len; i++){
        b[i] = 0.0;
    }
}

void gemv(T* A, T* x, T* b){
    T tmp;
    for(int i=0; i<rows; i++){
        tmp = 0;
        for(int j=0; j<cols; j++){
            tmp += A[i*cols+j] * x[j];
        }
        b[i] = tmp;
    }
}

void print_vector(T* v, uint32_t len){
    printf("\n");
    for(int i=0; i<len; i++){
        printf("%lf ", v[i]);
        if(i%49==0){
            printf("\n");
        }
    }
}

T sum(T* v, uint32_t len){
    T sum = 0.0;
    for(int i=0; i<len; i++){
        sum += v[i];
    }
    return sum;
}

int main(int argc, char *argv[]){

    struct dpu_set_t set, dpu;
    uint32_t num_dpus;

    //DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &set));
    DPU_ASSERT(dpu_alloc(3, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(set, &num_dpus));


    T *A = (T*)malloc_split_aligned(rows, cols*sizeof(T), num_dpus);
    T *x = (T*)malloc(cols*sizeof(T));
    T *b = (T*)malloc(rows*sizeof(T));
    make_matrix(A);
    make_input_vector(x);
    make_zero_vector(b, rows);
    gemv(A, x, b);
    T sum_x = sum(x, cols);
    T sum_Ax = sum(b, rows);

    Timer timer;
    start(&timer, 0, 0);
    start(&timer, 5, 0);

    
    uint32_t data_offset = host_split_to_dpu(set, A, rows, cols*sizeof(T), num_dpus, 0);
    uint32_t end_offset =  host_broadcast_to_dpu(set, x, 1, cols*sizeof(T), data_offset);
    printf("end of data transfer\n");

    // preparing and parsing argument
    dpu_arguments_t* input_args = (dpu_arguments_t*) malloc(num_dpus * sizeof(dpu_arguments_t));

    for(int i=0; i<num_dpus; i++){
     input_args[i].input_start_offset = 0;
     input_args[i].input_type_size = cols*sizeof(T);
     input_args[i].data_start_offset = data_offset;
     input_args[i].data_len = 1;
     input_args[i].data_type_size = cols*sizeof(T);
     input_args[i].end_offset = end_offset;
     input_args[i].table_type_size = sizeof(T);
     input_args[i].table_len = 0;
    }

    prepare_input_len_and_parse_args(set, input_args, rows, cols*sizeof(T), num_dpus);
    uint32_t* output_lens = (uint32_t*)malloc(num_dpus*sizeof(uint32_t));
    for(int i=0; i<num_dpus; i++){
        output_lens[i] = input_args[i].input_len;
    }
    stop(&timer, 0);

    // lanch dpu codes
    start(&timer, 1, 0);
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    stop(&timer, 1);

    if(print_info){
      DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
      }
    }

    start(&timer, 2, 0);
    T* res = (T*)gather_to_host(set, output_lens, sizeof(T), end_offset, num_dpus);
    stop(&timer, 2);
    stop(&timer, 5);

    printf("the total time with timing consumed is (ms): ");
    print(&timer, 5, 1);
    printf("\n");
    printf("initial CPU-DPU input transfer (ms): ");
	print(&timer, 0, 1);
    printf("\n");
	printf("DPU Kernel Time (ms): ");
	print(&timer, 1, 1);
    printf("\n");
    printf("DPU-CPU Time (ms): ");
	print(&timer, 2, 1);
    printf("\n");

    printf("the correct sum is %f \n", sum_Ax);
    T s = sum(res, rows);
    printf("the calculated sum is %f \n", s);


    if(s == sum_Ax){
        printf("the result is correct\n");
    }
    else{
        printf("the result does not match cpu\n");
    }
    
    return 0;
}