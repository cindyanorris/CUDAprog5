CC = /usr/local/cuda-8.0//bin/nvcc
GENCODE_FLAGS = -arch=sm_30
#If you mess with these flags, be sure to restore them
#when done.
CC_FLAGS = -c --compiler-options -Wall,-Wextra,-O3,-m64
NVCCFLAGS = -m64 

wheresWaldo: wheresWaldo.o gpuFindWaldo.o gpuFindWaldoS.o 
	$(CC) $(GENCODE_FLAGS) wheresWaldo.o gpuFindWaldo.o gpuFindWaldoS.o -o wheresWaldo

wheresWaldo.o: wheresWaldo.cu CHECK.h defs.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) wheresWaldo.cu -o wheresWaldo.o

gpuFindWaldo.o: gpuFindWaldo.cu CHECK.h defs.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) gpuFindWaldo.cu -o gpuFindWaldo.o

gpuFindWaldoS.o: gpuFindWaldoS.cu CHECK.h defs.h 
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) gpuFindWaldoS.cu -o gpuFindWaldoS.o

clean:
	rm wheresWaldo *.o
