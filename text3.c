#include <stdio.h>
#include <mpi.h> 

void print_grid_to_file(char filename[50], double a[], int nx, int size, int rank);

int main(int argc, char **argv) {
	int i;
	int rank, size;
	int a[10];
	char filename = "textfile.txt";
	MPI_Init( &argc, &argv);
	MPI_Comm_size( &size);
	MPI_Comm_rank(&rank);

	for (i = 0; i < 10; i++) {
		a[i] = 0;
	}

	a[rank] = rank;

	parrallel_print_grid_to_file(filename, a, 10);

	MPI_Finalize();
	return 0;
}


void print_grid_to_file(char filename[50], double a[], int nx, int size, int rank) {
  int i,j;
  
  fp = fopen(filename, "w");
  if ( !fp ){
    fprintf(stderr, "Error: can't open file %s\n",fname);
    exit(4);
  }


  for(j=0; j < size; j++){
    for(i=0; i<nx+2; i++){
      fprintf(fp, "%lf ",x[i][j]);
      }
    fprintf(fp, "\n");
  }
  fclose(fp);
}


