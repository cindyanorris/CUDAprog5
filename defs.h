#define idx(ROW, COL, N) ((COL) + ((ROW) * (N))) 

typedef struct
{
   int * indices;  //holds the row and column indices for waldo locations
   int count;  //number of rows and columns stored in the indices array (2 * number of waldos)
   int size;   //size of the indices array (size >= count)
} locationType;

float gpuFindWaldo(unsigned char * map, int N, locationType * gpufound);

float gpuFindWaldoS(unsigned char * map, int N, locationType * gpufound);

