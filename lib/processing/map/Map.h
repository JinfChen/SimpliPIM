#ifndef MAP_H
#define MAP_H
#include "MapArgs.h"
#include "../ProcessingHelperHost.h"
#include "../../management/Management.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dpu.h>
void table_map(const char* src_name, const char* dest_name, uint32_t outputs, uint32_t output_type, handle_t* binary_handle, smalltable_management_t* table_management, uint32_t info);
#endif 