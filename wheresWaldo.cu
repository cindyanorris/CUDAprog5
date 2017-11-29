#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "defs.h"

//prototypes for functions in this file
void initMap(unsigned char * map, int N, int maxWaldos, locationType * waldos);
void compare(const char * processor, locationType * placed, locationType * found);
float cpuFindWaldo(unsigned char * map, int N, locationType * waldos);
void printMap(unsigned char * map, int N);
bool isWaldo(unsigned char * map, int row, int col, int N);
void placeWaldoOnMap(unsigned char * map, int N, int rowloc, int colloc);
void putWaldoInArray(locationType * waldos, int row, int col);
bool noDoubleUs(unsigned char * map, int row, int col, int N);
void addWaldos(unsigned char * map, int N, int numWaldos);
void usage();
void findWaldoTests();
void findWaldoSTests();

#define NUMTESTS 4
typedef struct
{
   int N;                   //number of rows and cols of the map
   float waldoSpeedupGoal;  //minimum speedup that you should aim for find Waldo tests
   int maxWaldos;           //maximum number of Waldos for the find Waldos tests
   float waldosSpeedupGoal; //minimum speedup that you should aim for find Waldos tests
} testType;

testType tests[NUMTESTS] = {{1 << 8, 15.0, 20, 1.2}, 
                            {1 << 10, 140.0, 25, 6.5},
                            {1 << 12, 190.0, 30, 12.5},
                            {1 << 14, 200.0, 35, 13.0}}; 
/*
   driver for the find waldo program.  
   The main calls the functions to perform the tests
   specified in the tests array.

   @param argc - count of arguments in argv array
               - should be exactly 2
   @param argv - command line arguments
                 argv[1] is "-a" perform all tests
                 argv[1] is "-1" perform find one waldo tests
                 argv[1] is "-n" perform find all waldos tests
 */
int main(int argc, char * argv[])
{
   time_t t;
   //seed the random number generator used for
   //generating waldos
   srand((unsigned) time(&t));
 
   //user needs to enter either -a, -1 or -n 
   if (argc < 2) usage();

   //perform all tests
   if (strcmp(argv[1], "-a") == 0) 
   {
      findWaldoTests();
      findWaldoSTests();
   }
   else if (strcmp(argv[1], "-1") == 0)
      //perform find waldo tests only
      findWaldoTests();
   else if (strcmp(argv[1], "-n") == 0)
      //perform find waldoS tests only
      findWaldoSTests();
   else
      usage();
}
/*
   prints usage info and exits.  This is called by the
   main if the user doesn't enter the proper command
   line argument.
 */
void usage()
{
   printf("Usage: wheresWaldo -a|-1|-n\n\n");
   printf("      -a run all tests\n"); 
   printf("      -1 run find one Waldo tests\n"); 
   printf("      -n run find all Waldos tests\n"); 
   exit(1);
}

/*
   Performs the find waldo tests that search for one waldo
   in a two-d array of characters.  A waldo is represented by
   a square of four 'W's.  For example,
   WW
   WW
   is a waldo.  In these tests, a single waldo is embedded in
   a two-d array of characters where each character is a W or 
   a space.  For example, the 8 by 8 array of characters below 
   contains one waldo at position (0, 4). The position of waldo 
   is the position of its top left character. 
W W WW W
WW WWW W
W W W W
WW WW WW
 WW WW W
W   W  W
W WWW  W
W W W WW
 */
void findWaldoTests()
{
   int i, N, maxWaldos;
   float cpuTime;
   float gpuTime;
   float speedup;

   //declare locationType objects to hold the
   //location of the Waldo
   locationType placed;   //where waldo was placed
   locationType cpufound; //where waldo was found by cpu
   locationType gpufound; //where waldo was found by gpu

   printf("\nFind Waldo Tests\n");
   printf("%10s\t%8s\t%9s\t%8s\t%8s\n", 
         "N*N", "CPU ms", "GPU ms", "Speedup", "Goal");
   for (i = 0; i < NUMTESTS; i++)
   {
      N = tests[i].N; 
      maxWaldos = 1;
      //implement two-D array of size N by N with one-D array
      unsigned char * map = (unsigned char *) malloc(sizeof(unsigned char) * N * N);
      placed.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      cpufound.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      gpufound.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      placed.count = cpufound.count = gpufound.count = 0;
      placed.size = cpufound.size = gpufound.size = maxWaldos * 2;

      //put one waldo on the map
      initMap(map, N, 1, &placed);
  
      //time how long it takes for the cpu and gpu to find waldo
      cpuTime = cpuFindWaldo(map, N, &cpufound);      
      gpuTime = gpuFindWaldo(map, N, &gpufound);      

      //make sure the cpu produced the correct results
      compare("CPU", &placed, &cpufound);
      //make sure the gpu produced the correct results
      compare("GPU", &placed, &gpufound);

      speedup = cpuTime / gpuTime;

      //print the output
      printf("%10d\t%9.4f\t%8.4f\t%8.4f\t%8.1f\n", 
            N*N, cpuTime, gpuTime, 
            speedup, tests[i].waldoSpeedupGoal);

      //free the dynamically allocated data
      free(map);
      free(placed.indices);
      free(cpufound.indices);
      free(gpufound.indices);
   }
}   

/*
   Performs the find waldo tests that search for N waldos
   in a two-d array of characters.  A waldo is represented by
   a square of four 'W's.  For example,
   WW
   WW
   is a waldo.  In these tests, multiple waldos are embedded in
   a two-d array of characters where each character is a 'W'
   or a space.  For example, the 8 by 8 array of characters below 
   contains two waldo at positions (0, 4) and (1, 6). The 
   position of waldo is the position of its top left character. 
W W WW W
WW WWWWW
W W W WW
WW WW  W
 WW WW W
W   W  W
W WWW  W
W W W WW
*/
void findWaldoSTests()
{
   int i, N, maxWaldos; 
   float cpuTime;
   float gpuTime;
   float speedup;

   //declare locationType objects to hold the
   //locations of the Waldo
   locationType placed;   //where waldos were placed
   locationType cpufound; //where waldos were found by cpu
   locationType gpufound; //where waldos were found by gpu
   printf("\nFind Waldos Tests\n");
   printf("%10s\t%8s\t%9s\t%8s\t%8s\n", 
         "N*N", "CPU ms", "GPU ms", "Speedup", "Goal");
   for (i = 0; i < NUMTESTS; i++)
   {
      //test specifies the size of the map and the maximum number
      //of waldos on it
      N = tests[i].N; 
      maxWaldos = tests[i].maxWaldos;

      //implement two-D array of size N by N with one-D array
      unsigned char * map = (unsigned char *) malloc(sizeof(unsigned char) * N * N);

      //initialize the locationType objects
      placed.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      cpufound.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      gpufound.indices = (int *) malloc(sizeof(int) * maxWaldos * 2);
      placed.count = cpufound.count = gpufound.count = 0;
      placed.size = cpufound.size = gpufound.size = maxWaldos * 2;

      //initialize the map with up to maxWaldos waldos
      initMap(map, N, maxWaldos, &placed);

      //time how long it takes the CPU to find the waldos
      cpuTime = cpuFindWaldo(map, N, &cpufound);      

      //time how long it takes the GPU to find the waldos
      gpuTime = gpuFindWaldoS(map, N, &gpufound);      
      speedup = cpuTime / gpuTime;

      //make sure the CPU produced the correct results
      compare("CPU", &placed, &cpufound);
      //make sure the GPU produced the correct results
      compare("GPU", &placed, &gpufound);

      //print the output
      printf("%10d\t%9.4f\t%8.4f\t%8.4f\t%8.1f\n", 
            N*N, cpuTime, gpuTime, 
            speedup, tests[i].waldosSpeedupGoal);

      //free the dynamically allocated data
      free(map);
      free(placed.indices);
      free(cpufound.indices);
      free(gpufound.indices);
   }
}   


/*
   CPU function to find the waldos in the map.  A waldo is represented
   by a square of Ws.  The location of the waldo is the location of the
   top left corner of the square.

   @param map - one dimensional array to implement the 2D N by N array
                containing the waldos
   @param N - size of 1 dimension
   @param cpufound - struct that should be filled with the locations
                     of the waldos
          cpu->indices - filled with row and col of each waldo
                         waldo positions are added to the array in the
                         order of row then column
          cpu->size - size of indices array
          cpu->count - number of elements in the array
                       2 * number of waldos at end
 */
float cpuFindWaldo(unsigned char * map, int N, locationType * cpufound)
{
   cudaEvent_t start_cpu, stop_cpu;
   float cpuMsecTime = -1;
   int row, col, i = 0;

   //time how long it takes to find the waldos
   CHECK(cudaEventCreate(&start_cpu));
   CHECK(cudaEventCreate(&stop_cpu));
   CHECK(cudaEventRecord(start_cpu));

   for (row = 0; row < N; row++)
   {
      for (col = 0; col < N; col++)
      {
         if (isWaldo(map, row, col, N))
         {
            if (i == cpufound->size)
            {
               printf("No room in waldo array (size = %d) for waldo.\n", 
                     cpufound->size);
               exit(1);
            }
            //store waldo's location in the indices array
            cpufound->indices[i] = row;
            cpufound->indices[i + 1] = col;
            i += 2;
         }
      }
   } 
   cpufound->count = i;

   CHECK(cudaEventRecord(stop_cpu));
   CHECK(cudaEventSynchronize(stop_cpu));
   CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
   return cpuMsecTime;
}

/* 
   compares two lists of waldos to make sure they are the same

   @param processor - string used for printing error message
   @param placed - structure that indicates where waldos were placed
   @param found - structure that indicates where waldos were found
 */
void compare(const char * processor, locationType * placed, locationType * found)
{
   int i;
   bool bad = false;
   //number of elements in the indices arrays should be the same
   if (placed->count != found->count)
   {
      printf("%s error: %d waldos were placed. %d waldos were found.\n",
            processor, placed->count/2, found->count/2);
      bad = true;
   }
   //the waldo locations should match and be in the same order
   for (i = 0; bad == false && i < placed->count; i+= 2)
   {
      if (placed->indices[i] != found->indices[i] || 
            placed->indices[i+1] != found->indices[i+1])
      {
         bad = true;
      }
   }
   if (bad)
   {
      //print an error message and the locations if they don't match
      printf("%s error: locations don't match.\n", processor);
      for (i = 0; i < placed->count; i+= 2)
      {
         printf("Placed: (%d, %d)\n", 
               placed->indices[i], placed->indices[i+1]);
      }
      for (i = 0; i < found->count; i+= 2)
      {
         printf("Found: (%d, %d)\n",
               found->indices[i], found->indices[i+1]);
      }
   }
}

/*
   Checks to make sure there are no Ws in all of the elements
   that surround the element (row, col) in the map array. This
   is used to make sure that waldos won't be added to the
   map that are adjacent to each other.

   @param map - 1D array that simulates N by N array of chars
   @param row - potential row position for a waldo
   @param row - potential col position for a waldo
   @param N - size of one dimension
*/
bool noDoubleUs(unsigned char * map, int row, int col, int N)
{
   int i, j;
   for (i = max(0, row - 1); i < min(row + 3, N); i++)
   {
      for (j = max(0, col - 1); j < min(col + 3, N); j++)
      {
         if (map[idx(i, j, N)] == 'W') return false;
      }
   }
   return true;  //no W in the 4 by 4 square
}

/*
   Returns true if there is a waldo in position (row, col)
   in the map.

   @param map - one dimensional array that implements a
                two dimensional array of size N by N
   @param row - row number of possible waldo 
   @param col - col number of possbilel waldo 
   @param N - size of one dimension
*/
bool isWaldo(unsigned char * map, int row, int col, int N)
{
   if (row < 0 || col < 0) return false;
   if (row + 1 == N || col + 1 == N) return false;
   if (map[idx(row, col, N)] == 'W' &&
         map[idx(row, col + 1, N)] == 'W' && 
         map[idx(row + 1, col, N)] == 'W' && 
         map[idx(row + 1, col + 1, N)] == 'W') 
   {
      return true;
   }
   return false;
}

/*
   Fills in the map with a waldo.  A waldo is a square of Ws. The top left
   W is in position (rowloc, colloc).  Thus this function puts Ws in
   positions (rowloc, colloc), (rowloc, colloc+1), (rowloc+1, colloc) and
   (rowloc+1, colloc+1)
 
   @param map - one dimensional array that implements a
                two dimensional array of size N by N
   @param rowloc - row number of added waldo 
   @param colloc - col number of added waldo 
   @param N - size of one dimension
*/
void placeWaldoOnMap(unsigned char * map, int N, int rowloc, int colloc)
{
   int row, col;
   for (row = rowloc; row >= 0 && row - 2 < rowloc; row++)
   {
      for (col = colloc; col >= 0 && col - 2 < colloc; col++)
      {
         map[idx(row, col, N)] = 'W';
      } 
   }
}

/*
   Adds numWaldos to the map and fills the indices array within
   the locationType struct with the row and col position of each
   added waldo. The waldos that are added will not be next to each
   other. 

   @param map - one dimensional array that implements a
                two dimensional array of size N by N
   @param N - size of one dimension
   @param numWaldos - number of waldos to add
   @param placed - structure that will contain the
                   indices of the waldos added to the map
          placed->indices - filled with row and col of each waldo
          placed->size - size of indices array
          placed->count - number of elements in the array
                          2 * number of waldos at end
*/
void addWaldos(unsigned char * map, int N, int numWaldos, 
      locationType * placed)
{
   int rowloc, colloc;
   int i = 0;
   //repeat until numWaldos are added
   while (i < numWaldos)
   {
      rowloc = rand() % (N - 1);
      colloc = rand() % (N - 1);
      if (noDoubleUs(map, rowloc, colloc, N))
      {
         placeWaldoOnMap(map, N, rowloc, colloc);
         putWaldoInArray(placed, rowloc, colloc);
         i++;
      }
   }
}

/*
   Adds a waldo to the indices array within the locationType
   object. The waldo is added so that the order of the waldos
   within the indices array is first by row and then by column.
   For example, if the waldos are at positions (20, 0), (5, 2)
   and (5, 15), the row, cols will be placed in the array in 
   the order: 5, 2, 5, 15, 20, 0
   
   @param placed - structure that will contain the
                   indices of the waldos added to the map
          placed->indices - filled with row and col of each waldo
          placed->size - size of indices array
          placed->count - number of elements in the array
   @param row - row position of added waldo
   @param col - col position of added waldo
*/
void putWaldoInArray(locationType * placed, int row, int col)
{
   if (placed->count == placed->size)
   {
      printf("Error: placed array is full.\n"); 
      exit(1);
   }
   int j;
   //insert the new waldo into the indices array
   //maintaining sorted order
   for (j = placed->count - 2; j >= 0; j -= 2)
   {
      if (row > placed->indices[j] || 
         (row == placed->indices[j] && col > placed->indices[j + 1]))
         break;
      placed->indices[j + 2] = placed->indices[j];
      placed->indices[j + 3] = placed->indices[j + 1];
   }
   placed->indices[j + 2] = row;
   placed->indices[j + 3] = col;
   placed->count += 2;
}

/*
   Initializes the map with waldos and extra Ws that aren't part of a waldo.

   @param map - one dimensional array that implements a
                two dimensional array of size N by N
   @param N - size of one dimension
   @param maxWaldos - maximum number of waldos that can be added
   @param placed - structure that will contain the
                   indices of the waldos added to the map
          placed->indices - filled with row and col of each waldo
          placed->size - size of indices array
          placed->count - number of elements in the array
*/

void initMap(unsigned char * map, int N, int maxWaldos, locationType * waldos)
{
   int row, col, numWaldos = 1; 
   //initialize the map to spaces
   memset(map, ' ', sizeof(char) * N * N);

   //randomly generate the number of waldos to be added to the map
   if (maxWaldos != 1) numWaldos = rand() % (maxWaldos + 1);

   //add the waldos
   addWaldos(map, N, numWaldos, waldos);

   //add a bunch of random Ws, but make sure doing so doesn't
   //add another waldo
   for (row = 0; row < N; row++)
   {
      for (col = 0; col < N; col++)
      {
         //if this isn't one of the Waldo squares
         if (map[idx(row, col, N)] != 'W')
         {
            if (rand() % 2 == 1)  // change to a 'W'
            {
               //add the 'W' and then remove it if a Waldo was added
               map[idx(row, col, N)] = 'W';  
               if (isWaldo(map, row-1, col-1, N)) map[idx(row, col, N)] = ' ';
               if (isWaldo(map, row-1, col, N)) map[idx(row, col, N)] = ' ';
               if (isWaldo(map, row, col-1, N)) map[idx(row, col, N)] = ' ';
               if (isWaldo(map, row, col, N)) map[idx(row, col, N)] = ' ';
            }
         }
      }
   }
}

/* 
   Helper function to print the map
*/
void printMap(unsigned char * map, int N)
{
   int row, col;
   printf("   ");
   for (row = 0; row < N; row++) printf("%2d", row % 10);
   for (row = 0; row < N; row++)
   {
      printf("\n%2d: ", row);
      for (col = 0; col < N; col++)
      {
         printf("%2c", map[idx(row, col, N)]);
      }
   }
}

