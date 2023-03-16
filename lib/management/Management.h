#ifndef MANAGEMENT_H
#define MANAGEMENT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dpu.h>

#include "SmallTableInit.h"
#include "../processing/gen_red/GenRedArgs.h"
#include "../processing/map/MapArgs.h"
#include "../processing/zip/ZipArgs.h"

typedef struct { 
   char* name;
   uint32_t start;
   uint32_t end;
   uint64_t len;
   uint32_t* lens_each_dpu;
   uint32_t table_type_size;

   // fields handling virtual zip
   uint32_t is_virtual_zipped;
   uint32_t start1;
   uint32_t start2;
   uint32_t end1;
   uint32_t end2;
   uint32_t type1;
   uint32_t type2;
} table_host_t;

typedef struct { 
   uint32_t curr_space;
   uint32_t num_tables;
   table_host_t** tables;
   struct dpu_set_t set;
   uint32_t num_dpus;
   zip_arguments_t* zip_args;
   map_arguments_t* map_args;
   gen_red_arguments_t* red_args;
} smalltable_management_t;


smalltable_management_t* table_management_init(uint32_t num_dpus);
void add_table(table_host_t* table, smalltable_management_t* management);
uint32_t contains_table(const char* name, smalltable_management_t* management);
table_host_t* lookup_table(const char* name, smalltable_management_t* management);
void free_table(const char* name, smalltable_management_t* management);
uint32_t max_len_dpu(uint32_t num_dpus, table_host_t* table);

#endif 