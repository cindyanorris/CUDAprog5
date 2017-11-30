#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "defs.h"

/* 
   Uses the GPU to perform a find Waldo on the input map.
   Fills the locationType object with the location of the
   waldo that it found.  

   @param map - one dimensional array to implement the 2D N by N array
                containing the waldos
   @param N - size of 1 dimension
   @param gpufound - struct that should be filled with the locations
                     of the waldos
          gpu->indices - filled with row and col of each waldo
                         waldo positions are added to the array in the
                         order of row then column
          gpu->size - size of indices array 
          gpu->count - number of elements in the array
                       2 * number of waldos at end

   In this case, the number of waldos in the map will be exactly one,
   thus the gpuFindWaldo function will set gpufound->count to 2,
   gpufound->indices[0] to the row position of the waldo, and
   gpufound->indices[1] to the col position of the waldo

   @return amount of time it takes to find waldo in millisecond
*/
float gpuFindWaldo(unsigned char * map, int N, locationType * gpufound)
{
    unsigned char * dMap; 

    //create input array for GPU
    CHECK(cudaMalloc((void **)&dMap, sizeof(unsigned char) * N * N));
    CHECK(cudaMemcpy(dMap, map, sizeof(unsigned char) * N * N, 
               cudaMemcpyHostToDevice));

    //You may cudaMalloc some more space here that you will need
    //before the timing begins.

    float gpuMsecTime = -1;
    cudaEvent_t start_cpu, stop_cpu;
    //start the timing
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    //Write the findWaldo function. 
    //Before exiting gpuFindWaldo, your code
    //will need to have filled the gpufound struct.
    //You can either do that here or in your findWaldo
    //function.

    //findWaldo(....);

    //stop the timing
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_cpu, stop_cpu));
   
    CHECK(cudaFree(dMap));

    //free any other spaces you allocated

    return gpuMsecTime;
}

