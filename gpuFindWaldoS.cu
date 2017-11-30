#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "defs.h"

/* 
   Uses the GPU to perform a find all of the Waldos in the input map.
   Fills the locationType object with the locations of the
   waldos that it finds.  

   @param map - one dimensional array to implement the 2D N by N array
                containing the waldos
   @param N - size of 1 dimension
   @param gpufound - struct that should be filled with the locations
                     of the waldos
          gpufound->indices - filled with row and col of each waldo
                              waldo positions are added to the array in the
                              order of row then column
          gpufound->size - size of indices array
          gpufound->count - number of elements in the array
                            2 * number of waldos at end

   For example, if the waldos are in positions (3, 20), (10, 40),
   (2, 5), (3, 60) then the indices array will be filled as follows:
   gpufound->indices: 2, 5, 3, 20, 3, 60, 10, 40
   Note that the row and col pairs are in consecutive elements in the
   array and the array is ordered first by row (e.g., 2, 3, 3, 10) and then
   by column (e.g., 3, 20 comes before 3, 60)
   gpufound->count will be set to 8 since the indices array will have
   8 elements at the end.

   @return time it takes to find the waldos in milliseconds
*/
float gpuFindWaldoS(unsigned char * map, int N, locationType * gpufound)
{
    unsigned char * dMap; 

    //create input array for GPU
    CHECK(cudaMalloc((void **)&dMap, sizeof(unsigned char) * N * N));
    CHECK(cudaMemcpy(dMap, map, sizeof(unsigned char) * N * N, 
               cudaMemcpyHostToDevice));

    //You may want to cudaMalloc more space here that you will
    //need before the timing begins

    float gpuMsecTime = -1;
    cudaEvent_t start_cpu, stop_cpu;

    //start the timing
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    //Write the findWaldoS function. 
    //Before exiting gpuFindWaldoS, your code
    //will need to have filled the gpufound struct.
    //You can either do that here or in your findWaldoS
    //function.

    //findWaldoS(...);

    //stop the timing
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_cpu, stop_cpu));
   
    //release the space for the GPU arrays
    CHECK(cudaFree(dMap));

    //free any other space you allocated

    return gpuMsecTime;
}

