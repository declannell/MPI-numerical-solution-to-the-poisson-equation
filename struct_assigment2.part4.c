#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobi.h"

#define maxit 10000

#include "decomp1d.h"

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);
void MPE_decomp2d(int nx, int dim[2], int myid, int mycoords[2], int s[2], int e[2]);
void twodinit_basic(double a[][maxn], double b[][maxn], double f[][maxn], int nx, int ny, int s[2], int e[2]);
void print_full_grid(double x[][maxn], int nx);
void print_in_order(double x[][maxn], MPI_Comm comm, int nx);
void  print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);
void analytic_grid(double analytic[][maxn], int nx, int ny);
void gather_grid( double a[][maxn], int nx, int size, int myid, MPI_Comm comm);

typedef struct 
{
  int ex;
  int ey;
  int sx;
  int sy;
} Start_End_Coords;



int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  double analytic[maxn][maxn];
  int nx, ny;
  int myid, nprocs;
  /* MPI_Status status; */
  int nbrleft, nbrright, nbrup, nbrdown;
  int it;
  double glob_diff, global_analytic_diff;
  double ldiff;
  double t1, t2;
  double tol=1.0E-11;
  MPI_Comm cartcomm2d;
  int ndim = 2;
  
  int dim[2] = {0, 0};
  int period[2] = {0, 0};
  int reorder = 0;
  int mycoords[2] = {0, 0};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


  if( myid == 0 ){
    /* set the size of the problem */
    if(argc > 2){
      fprintf(stderr,"---->Usage: mpirun -np <nproc> %s <nx>\n",argv[0]);
      fprintf(stderr,"---->(for this code nx=ny)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if( argc == 2 ){
      nx = atoi(argv[1]);
    }
    if( argc == 1 ){
      nx=15;
    }

    if( nx > maxn-2 ){
      fprintf(stderr,"grid size too large\n");
      exit(1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  ny = nx;

  // This function will determine the number of processors to place in each dimension
  MPI_Dims_create(nprocs, 2, dim);
  if (myid == 0) {
    printf("The number of processors in the first dimension is %d. The number of processors in the second dimesion is %d\n", dim[0], dim[1]);
  }

  init_full_grids(a, b, f);
  // This creates the communicator with a 2d cartesian topology.
  MPI_Cart_create(MPI_COMM_WORLD, ndim, dim, period, reorder, &cartcomm2d);
  // This detremines the coordinates of each process within the 2d topology.
  MPI_Cart_coords(cartcomm2d, myid, ndim, mycoords);
  //This function determines the nearest neighbours in each direction.
  MPI_Cart_shift(cartcomm2d, 0, 1, &nbrleft, &nbrright);

  MPI_Cart_shift(cartcomm2d, 1, 1, &nbrdown, &nbrup);

  MPE_decomp2d(nx, dim, myid, mycoords, s, e);
  printf(" myid is %d,(%d, %d), nbrup = %d, nbrdown = %d, nbrright = %d, nbrleft = %d,s = ( %d, %d), e = (%d,%d)\n", myid, mycoords[0], mycoords[1], nbrup, nbrdown, nbrright, nbrleft,  s[0], s[1], e[0], e[1]);  printf(" myid is %d,(%d, %d), nbrup = %d, nbrdown = %d, nbrright = %d, nbrleft = %d,s = ( %d, %d), e = (%d,%d)\n", myid, mycoords[0], mycoords[1], nbrup, nbrdown, nbrright, nbrleft,  s[0], s[1], e[0], e[1]);


  twodinit_basic(a, b, f, nx, ny, s, e);
  print_in_order(a, MPI_COMM_WORLD, nx);



  t1 = MPI_Wtime();

  glob_diff = 1000;
  for(it=0; it< maxit; it++){
    if( it == 0 ){
      printf("\n======> NB VERSION\n\n");
    }

    // update b using a 
    exchang3_2d(a, ny, s, e, MPI_COMM_WORLD, nbrleft, nbrright, nbrup, nbrdown);
    //print_in_order(a, MPI_COMM_WORLD, nx);
    sweep2d(a, f, nx, s, e, b);
    //print_in_order(b, MPI_COMM_WORLD, nx);
    // update a using b 


    exchang3_2d(b, ny, s, e, MPI_COMM_WORLD, nbrleft, nbrright, nbrup, nbrdown);
    //print_in_order(b, MPI_COMM_WORLD, nx);
    sweep2d(b, f, nx, s, e, a);
    //print_in_order(a, MPI_COMM_WORLD, nx);
    ldiff = griddiff_2d(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(myid==0 && it%10==0){
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n",myid, ldiff, glob_diff);
    }
    if( glob_diff < tol ){
      if(myid==0){
	printf("iterative solve converged\n");
      }
      break;
    }
  //Note you should use a as the converged grid, and not b 
  }

  t2=MPI_Wtime();
  printf("DONE! (it: %d)\n",it);


  if( myid == 0 ){
    if( it == maxit ){
      fprintf(stderr,"Failed to converge\n");
    }
    printf("Run took %lf s\n",t2-t1);
  }
/*
  //print_in_order(a, MPI_COMM_WORLD, nx);
  if( nprocs == 1  ){
    print_grid_to_file("gridnb", a,  nx, ny);
    print_full_grid(a, nx);
  }

  analytic_grid(analytic, nx, ny);
  if (myid == 0) {
    printf("The analytic grid is\n");
    print_full_grid(analytic, nx);
  }

  ldiff = griddiff(analytic, a, nx, s, e);// I change the griddiff function to return actually difference and not the difference squared
  MPI_Allreduce(&ldiff, &global_analytic_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (myid == 0) {
    printf("The global difference to the analytic solution on a grid size of %d, is %f\n", nx, global_analytic_diff);
  }

  gather_grid( a, nx, nprocs, myid,  MPI_COMM_WORLD);
  if (myid == 0) {
	printf(" The converged grid on rank 0 is \n");
	print_full_grid(a, nx);
  }
*/
 MPI_Finalize();
 return 0;
}

void MPE_decomp2d(int nx, int dim[2], int myid, int mycoords[2], int s[2], int e[2]) {
   /* We use the 1d decomp for each dimension. s and e beome arrays with two entries, one for each dimension. More detailed explaination in pdf with assignment. 
   We need an effective rank for each row or column of processors. This allows us to use the 1d decomp for each row and the each column to detering the starting and end points in 
   each direction. This is hard coded for a 2d decomposition. */
   //this is the decomposition in the first dimension.


   int effective_1d_rank = (myid - mycoords[1]) / dim[1];
   //printf("For myid = %d, effective_1d_rank = %d, x direction\n " , myid, effective_1d_rank);
   MPE_Decomp1d(nx, dim[0], effective_1d_rank, s, e);

   //This is the y direction decomposition.
   effective_1d_rank = (myid) % dim[1];
   //printf("For myid = %d, effective_1d_rank = %d, y direction\n " , myid, effective_1d_rank);
   MPE_Decomp1d(nx, dim[1], effective_1d_rank, &s[1], &e[1]);


}


void twodinit_basic(double a[][maxn], double b[][maxn], double f[][maxn], int nx, int ny, int s[2], int e[2])
{
  /*This function is  edited for the analytic solution to the poisson equation.*/

  int i,j;
  double x, y;


  /* set everything to 0 first */
  for(i = s[0] - 1; i <= e[0] + 1; i++){
    for(j = s[1] - 1; j <= e[1] + 1; j++){
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  if ( s[1] == 1) {
  // deal with boundaries. These  starting points only need to start at i = s[1] and end at e[1] to get the correct answer but I added the other boundaries for visualing the grids.
    for (i = s[0] - 1; i <= e[0] + 1; i++) {
      x = 1.0/((double)nx + 1.0) * i;
      a[i][0] = 0.0;
      b[i][0] = 0.0;
    }
  }

  if (e[1] == nx) {
    for ( i = s[0] - 1; i <= e[0] + 1; i++) {
      x = 1.0/((double)nx + 1.0) * i;
      a[i][nx+1] = 1.0/((x +1.0) * (x +1.0)+1.0);
      b[i][nx+1] = 1.0/((x +1.0) * (x +1.0)+1.0);
    }
  }
  if( s[0] == 1 ){
    printf("This occured\n");
    for(i = s[1] - 1; i <= e[1] + 1; i++){
      y = 1.0/((double)ny + 1.0) * i;
      a[0][i] = y / (y*y + 1.0);
      b[0][i] = y / (y*y + 1.0);
    }
  }


  if( e[0] == nx ){
    for(i = s[1] - 1; i <= e[1] + 1; i++){
      y = 1.0/((double)ny + 1.0) * i;
      a[nx+1][i] = y / (y*y + 4.0);
      b[nx+1][i] = y / (y*y + 4.0);
    }
  }

}

void init_full_grid(double g[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      g[i][j] = junkval;
    }
  }
}

/* set global a,b,f to initial arbitrarily chosen junk value */
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }

}

void print_full_grid(double x[][maxn], int nx)
{
  int i,j;
  for(j = nx + 1; j >= 0; j--){
    for(i = 0; i < nx + 2; i++){
      if(x[i][j] < 10000.0){
	printf("|%2.6lf| ",x[i][j]);
      } else {
	printf("%9.2lf ",x[i][j]);
      }
    }
    printf("\n");
  }

}

void print_in_order(double x[][maxn], MPI_Comm comm, int nx)
{
  int myid, size;
  int i;

  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);
  printf("Attempting to print in order\n");
  sleep(1);
  MPI_Barrier(comm);

  for(i=0; i<size; i++){
    if( i == myid ){
      printf("proc %d\n",myid);
      print_full_grid(x, nx);
    }
    fflush(stdout);
    usleep(500);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny)
{
  FILE *fp;
  int i,j;

  fp = fopen(fname, "w");
  if ( !fp ){
    fprintf(stderr, "Error: can't open file %s\n",fname);
    exit(4);
  }

  for(j=ny+1; j>=0; j--){
    for(i=0; i<nx+2; i++){
      fprintf(fp, "%lf ",x[i][j]);
      }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void analytic_grid(double analytic[][maxn], int nx, int ny) {
	/*This function will write the analytic solution to the problem in the 2D array grid*/
	int i, j;
	double x, y;
	for (i = 0; i < nx + 2; i++) {
		for (j = 0; j < nx + 2; j++) {
			x = 1.0 / ((double)nx + 1.0) * i;
			y = 1.0 / ((double)ny + 1.0) * j;
			analytic[i][j] = y / ((1.0 + x) * (1.0 +x) + y * y);
		}
	}
  }
/*
void gather_grid( double a[][maxn], int nx, int size, int myid, MPI_Comm comm) {
	int i, j;
	//these are so processor know sthe starting points of the other processors
	int s[12], e[12], num_cols[12]; // max number of proccessors on chuck is 12 so we set the max length of the arrays to 12.
	//num_cols is the number of columns on each processor so we need this for communication to processor 0.
        if (myid == 0){
	    if (size > 12) {
      		fprintf(stderr,"The number of processors is greater than 12 and hence the starting and end arrays are two small\n");
	    }
        }

 	for ( i = 0; i < size; i++) {
		MPE_Decomp1d(nx, size, i, (s + i), (e + i));
  		num_cols[i] = e[i] -s[i] + 1;
		//printf(" s[%d] = %d, e[%d] = %d num_cols[%d] = %d\n", i , s[i], i , e[i], i, num_cols[i]);
	}


        if (myid != 0) {
		for (j = 0; j < num_cols[myid]; j++) {
                	MPI_Send(&a[s[myid]+j][0], (nx + 2), MPI_DOUBLE, 0, myid * j + myid, comm);
        	}
	}

        if ( myid == 0) {
                for (i = 1; i < size; i++) {
			for  (j = 0; j < num_cols[i]; j++) {
                        	MPI_Recv(&a[s[i]+j][0], (nx + 2), MPI_DOUBLE, i, i * j + i, comm, MPI_STATUS_IGNORE);
                	}
       		 }
	}

	//This is for the right boundary
	if ( myid == size - 1) {
        	MPI_Send(&a[nx + 1], (nx + 2), MPI_DOUBLE, 0, 89, comm);
	}
	if (myid == 0) {
		MPI_Recv(&a[nx + 1], (nx + 2), MPI_DOUBLE, size -1, 89, comm, MPI_STATUS_IGNORE);
	}
}*/
