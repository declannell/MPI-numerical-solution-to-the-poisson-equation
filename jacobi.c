#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "poisson1d.h"
#include "jacobi.h"


void sweep1d(double a[][maxn], double f[][maxn], int nx,
	     int s, int e, double b[][maxn])
{
  double h;
  int i,j;

  h = 1.0/((double)(nx+1));

  for(i=s; i<=e; i++){
    for(j=1; j<nx+1; j++){
      b[i][j] = 0.25 * ( a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1]  - h*h*f[i][j] );
    }
  }
}


void sweep2d(double a[][maxn], double f[][maxn], int nx,
	     int s[2], int e[2], double b[][maxn])
{
  double h;
  int i,j;

  h = 1.0/((double)(nx+1));

  for (i = s[0]; i <= e[0]; i++) {
    for (j = s[1] ; j <= e[1] ; j++) {
      b[i][j] = 0.25 * ( a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1]  - h*h*f[i][j] );
    }
  }
}


/* sendrecv */
       /* int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, */
       /*      int dest, int sendtag, void *recvbuf, int recvcount, */
       /*      MPI_Datatype recvtype, int source, int recvtag, */
       /*      MPI_Comm comm, MPI_Status *status) */
void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright)
{

  MPI_Sendrecv(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, &x[s-1][1], nx, MPI_DOUBLE, nbrleft,
	       0, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, &x[e+1][1], nx, MPI_DOUBLE, nbrright,
	       1, comm, MPI_STATUS_IGNORE);

}


void exchang3_2d_not_sendrecv(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown, int mycoords[2])
{
  int num_y = e[1] - s[1] + 1; //this is the number of elemnts in the "y direction" of the grid.

  if(mycoords[0] % 2 == 0){

    MPI_Ssend(&x[e[0]][s[1]], num_y, MPI_DOUBLE, nbrright, 0, comm);

    MPI_Recv(&x[s[0] - 1][s[1]], num_y, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);

    MPI_Ssend(&x[s[0]][s[1]], num_y, MPI_DOUBLE, nbrleft, 1, comm);

    MPI_Recv(&x[e[0] + 1][s[1]], num_y, MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&x[s[0]-1][s[1]], num_y, MPI_DOUBLE, nbrleft, 0, comm, MPI_STATUS_IGNORE);

    MPI_Ssend(&x[e[0]][s[1]], num_y, MPI_DOUBLE, nbrright, 0, comm);

    MPI_Recv(&x[e[0] + 1][s[1]], num_y, MPI_DOUBLE, nbrright, 1, comm, MPI_STATUS_IGNORE);

    MPI_Ssend(&x[s[0]][s[1]], num_y, MPI_DOUBLE, nbrleft, 1, comm);

/*
  MPI_Datatype row_type;
  MPI_Type_vector(e[0] - s[0] + 1, 1, maxn, MPI_DOUBLE, &row_type); // We have to skip maxn as this is actually the size of the grid, not nx + 2.
  MPI_Type_commit(&row_type);

  MPI_Sendrecv(&x[s[0]][s[1]], 1 , row_type, nbrdown, 2, &x[s[0]][e[1]+1], 1, row_type, nbrup,
	       2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[s[0]][e[1]], 1, row_type, nbrup, 3, &x[s[0]][s[1] - 1], 1, row_type, nbrdown,
	       3, comm, MPI_STATUS_IGNORE);

*/
  }

}



void exchang3_2d(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown)
{

/*
  MPI_Sendrecv(&x[e[0]][s[1]], e[1] - s[1] + 1 , MPI_DOUBLE, nbrright, 0, &x[s[0]-1][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrleft,
	       0, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&x[s[0]][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrleft, 1, &x[e[0]+1][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrright,
	       1, comm, MPI_STATUS_IGNORE);
*/
  MPI_Datatype row_type;
  MPI_Type_vector(e[0] - s[0] + 1, 1, maxn, MPI_DOUBLE, &row_type); // We have to skip maxn as this is actually the size of the grid, not nx + 2.
  MPI_Type_commit(&row_type);

  MPI_Sendrecv(&x[s[0]][s[1]], 1 , row_type, nbrdown, 2, &x[s[0]][e[1]+1], 1, row_type, nbrup,
	       2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&x[s[0]][e[1]], 1, row_type, nbrup, 3, &x[s[0]][s[1] - 1], 1, row_type, nbrdown,
	       3, comm, MPI_STATUS_IGNORE);


}


void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	       int nbrleft, int nbrright)
{
  MPI_Request reqs[4];

  MPI_Irecv(&x[s-1][1], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&x[e+1][1], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&x[e][1],   nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&x[s][1],   nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);
  /* not doing anything useful here */

  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s, int e, double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright)
{
  MPI_Request req[4];
  MPI_Status status;
  int idx;
  double h;
  int i,j,k;

  int myid;
  MPI_Comm_rank(comm, &myid);

  h = 1.0/( (double)(nx+1) );

    /* int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, */
    /*               int source, int tag, MPI_Comm comm, MPI_Request *request); */
    /* int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, */
    /* 		  int tag, MPI_Comm comm, MPI_Request *request); */

  MPI_Irecv(&u[s-1][1], ny, MPI_DOUBLE, nbrleft, 1, comm, &req[0] );
  MPI_Irecv(&u[e+1][1], ny, MPI_DOUBLE, nbrright, 2, comm, &req[1] );

  MPI_Isend(&u[e][1], ny, MPI_DOUBLE, nbrright, 1, comm, &req[2]);
  MPI_Isend(&u[s][1], ny, MPI_DOUBLE, nbrleft, 2, comm, &req[3]);

  /* perform purely local updates (that don't need ghosts) */
  /* 2 cols or less means all are on processor boundary */
  if( e-s+1 > 2 ){
    for(i=s+1; i<e; i++){
      for(j=1; j<ny+1; j++){
	unew[i][j] = 0.25 * ( u[i-1][j] + u[i+1][j] + u[i][j+1] + u[i][j-1]  - h*h*f[i][j] );
      }
    }
  }

  /* perform updates in j dir only for boundary cols */
  for(j=1; j<ny+1; j++){
    unew[s][j] = 0.25 * ( u[s][j+1] + u[s][j-1]  - h*h*f[s][j] );
    unew[e][j] = 0.25 * ( u[e][j+1] + u[e][j-1]  - h*h*f[e][j] );
  }

  /* int MPI_Waitany(int count, MPI_Request array_of_requests[], */
  /*      int *index, MPI_Status *status) */
  for(k=0; k < 4; k++){
    MPI_Waitany(4, req, &idx, &status);

    /* idx 0, 1 are recvs */
    switch(idx){
    case 0:
      /* printf("myid: %d case idx 0: status.MPI_TAG: %d; status.MPI_SOURCE: %d (idx: %d)\n",myid,status.MPI_TAG, status.MPI_SOURCE,idx); */
      if( nbrleft != MPI_PROC_NULL &&
	  (status.MPI_TAG != 1 || status.MPI_SOURCE != nbrleft )){
	fprintf(stderr, "Error: I don't understand the world: (tag %d; source %d)\n",
		status.MPI_TAG, status.MPI_SOURCE);
	MPI_Abort(comm, 1);
      }

      /* left ghost update completed; update local leftmost column */
      for(j=1; j<ny+1; j++){
	unew[s][j] += 0.25 * ( u[s-1][j] );
      }
      break;
    case 1:
      /* printf("myid: %d case idx 1: status.MPI_TAG: %d; status.MPI_SOURCE: %d (idx: %d)\n",myid, status.MPI_TAG, status.MPI_SOURCE,idx); */
      if(nbrright != MPI_PROC_NULL &&
	 (status.MPI_TAG != 2 || status.MPI_SOURCE != nbrright )){
	fprintf(stderr, "Error: I don't understand the world: (tag %d; source %d)\n",
		status.MPI_TAG, status.MPI_SOURCE);
	MPI_Abort(comm, 1);
      }
      /* right ghost update completed; update local rightmost
	 column */
      for(j=1; j<ny+1; j++){
	unew[e][j] += 0.25 * ( u[e+1][j] );
      }
      break;
    default:
      break;
    }
  }
  /* splitting this off to take account of case of one column assigned
     to proc -- so left and right node neighbours are ghosts so both
     the recvs must be complete*/
  for(j=1; j<ny+1; j++){
    unew[s][j] += 0.25 * ( u[s+1][j] );
    unew[e][j] += 0.25 * ( u[e-1][j] );
  }

}


void nbxchange_and_sweep_2d(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s, int e, double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright, int nbrup, int nbrdown)
{
  MPI_Request req[16];
  MPI_Status status;
  int idx;
  double h;
  int i,j,k;

  int myid;
  MPI_Comm_rank(comm, &myid);

  //this assumes that the grid is a square
  h = 1.0/( (double)(nx+1) );

    /* int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, */
    /*               int source, int tag, MPI_Comm comm, MPI_Request *request); */
    /* int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, */
    /* 		  int tag, MPI_Comm comm, MPI_Request *request); */

  MPI_Irecv(&u[s-1][1], ny, MPI_DOUBLE, nbrleft, 1, comm, &req[0] );
  MPI_Irecv(&u[e+1][1], ny, MPI_DOUBLE, nbrright, 2, comm, &req[1] );

  MPI_Isend(&u[e][1], ny, MPI_DOUBLE, nbrright, 1, comm, &req[2]);
  MPI_Isend(&u[s][1], ny, MPI_DOUBLE, nbrleft, 2, comm, &req[3]);

  /* perform purely local updates (that don't need ghosts) */
  /* 2 cols or less means all are on processor boundary */
  if( e-s+1 > 2 ){
    for(i=s+1; i<e; i++){
      for(j=1; j<ny+1; j++){
	unew[i][j] = 0.25 * ( u[i-1][j] + u[i+1][j] + u[i][j+1] + u[i][j-1]  - h*h*f[i][j] );
      }
    }
  }

  /* perform updates in j dir only for boundary cols */
  for(j=1; j<ny+1; j++){
    unew[s][j] = 0.25 * ( u[s][j+1] + u[s][j-1]  - h*h*f[s][j] );
    unew[e][j] = 0.25 * ( u[e][j+1] + u[e][j-1]  - h*h*f[e][j] );
  }

  /* int MPI_Waitany(int count, MPI_Request array_of_requests[], */
  /*      int *index, MPI_Status *status) */
  for(k=0; k < 4; k++){
    MPI_Waitany(4, req, &idx, &status);

    /* idx 0, 1 are recvs */
    switch(idx){
    case 0:
      /* printf("myid: %d case idx 0: status.MPI_TAG: %d; status.MPI_SOURCE: %d (idx: %d)\n",myid,status.MPI_TAG, status.MPI_SOURCE,idx); */
      if( nbrleft != MPI_PROC_NULL &&
	  (status.MPI_TAG != 1 || status.MPI_SOURCE != nbrleft )){
	fprintf(stderr, "Error: I don't understand the world: (tag %d; source %d)\n",
		status.MPI_TAG, status.MPI_SOURCE);
	MPI_Abort(comm, 1);
      }

      /* left ghost update completed; update local leftmost column */
      for(j=1; j<ny+1; j++){
	unew[s][j] += 0.25 * ( u[s-1][j] );
      }
      break;
    case 1:
      /* printf("myid: %d case idx 1: status.MPI_TAG: %d; status.MPI_SOURCE: %d (idx: %d)\n",myid, status.MPI_TAG, status.MPI_SOURCE,idx); */
      if(nbrright != MPI_PROC_NULL &&
	 (status.MPI_TAG != 2 || status.MPI_SOURCE != nbrright )){
	fprintf(stderr, "Error: I don't understand the world: (tag %d; source %d)\n",
		status.MPI_TAG, status.MPI_SOURCE);
	MPI_Abort(comm, 1);
      }
      /* right ghost update completed; update local rightmost
	 column */
      for(j=1; j<ny+1; j++){
	unew[e][j] += 0.25 * ( u[e+1][j] );
      }
      break;
    default:
      break;
    }
  }
  /* splitting this off to take account of case of one column assigned
     to proc -- so left and right node neighbours are ghosts so both
     the recvs must be complete*/
  for(j=1; j<ny+1; j++){
    unew[s][j] += 0.25 * ( u[s+1][j] );
    unew[e][j] += 0.25 * ( u[e-1][j] );
  }

}

double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e)
{
  double sum;
  double tmp;
  int i, j;

  sum = 0.0;

  for(i=s; i <= e; i++){
    for(j=1;j<nx+1;j++){
      tmp = (a[i][j] - b[i][j]);
      sum = sum + sqrt(tmp*tmp);
    }
  }

  return sum;

}


double griddiff_2d(double a[][maxn], double b[][maxn], int nx, int s[2], int e[2])
{
  double sum;
  double tmp;
  int i, j;

  sum = 0.0;

  for (i = s[0]; i <= e[0]; i++) {
    for ( j = s[1]; j <= e[1]; j++) {
      tmp = (a[i][j] - b[i][j]);
      sum = sum + sqrt(tmp*tmp);
    }
  }

  return sum;

}
