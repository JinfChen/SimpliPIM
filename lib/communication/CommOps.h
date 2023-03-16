#ifndef COMMOPS_H
#define COMMOPS_H
#include "CommHelper.h"
#include "../management/Management.h"
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


void* malloc_scatter_aligned(uint64_t len, uint32_t type_size, smalltable_management_t* table_management);
void* malloc_reduce_aligned(uint32_t len, uint32_t type_size, smalltable_management_t* table_management);
void* malloc_broadcast_aligned(uint32_t len, uint32_t type_size, smalltable_management_t* table_management);
void small_table_scatter(char* const table_id, void* elements, uint64_t len, uint32_t type_size, uint32_t curr_offset, smalltable_management_t* table_management);
void* small_table_gather(char* const table_id, smalltable_management_t* table_management);
void small_table_broadcast(char* const table_id, void* elements, uint64_t len, uint32_t type_size, uint32_t curr_offset, smalltable_management_t* table_management);
#endif 