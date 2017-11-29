#define idx(ROW, COL, N) ((COL) + ((ROW) * (N))) 

typedef struct
{
   int * indices;  //holds the row and column indices for waldo locations
   int count;      //number of waldos on the map
   int size;       //size of the indices array
} locationType;

float gpuFindWaldo(unsigned char * map, int N, locationType * gpufound);

float gpuFindWaldoS(unsigned char * map, int N, locationType * gpufound);

