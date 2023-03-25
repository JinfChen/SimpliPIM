comm: host.c
	@mkdir -p bin
	dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=1 -o bin/dpu_binary dpu.c
	gcc --std=c99 -ldl -lm -fopenmp -O3 host.c -o bin/host ../../lib/processing/ProcessingHelperHost.c ../../lib/communication/CommHelper.c ../../lib/communication/CommOps.c ../../lib/management/SmallTableInit.c ../../lib/management/Management.c ../../lib/processing/gen_red/GenRed.c `dpu-pkg-config --cflags --libs dpu`