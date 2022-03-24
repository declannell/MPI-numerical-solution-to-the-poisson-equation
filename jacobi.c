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


void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright)
{

  MPI_Sendrecv(&x[e][1], nx, MPI_DOUBLE, nbrright, 0, &x[s-1][1], nx, MPI_DOUBLE, nbrleft,
	       0, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&x[s][1], nx, MPI_DOUBLE, nbrleft, 1, &x[e+1][1], nx, MPI_DOUBLE, nbrright,
	       1, comm, MPI_STATUS_IGNORE);

}

void exchang3_2d_nb_sendrecv(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown, int mycoords[2])
{

  if (mycoords[0] % 2 == 0) {
  MPI_Request reqs[4];

  MPI_Irecv(&x[s[0] - 1][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&x[e[0] + 1][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&x[e[0]][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&x[s[0]][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);

  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  } else {

  MPI_Request reqs1[4];

  MPI_Isend(&x[e[0]][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs1[0]);
  MPI_Isend(&x[s[0]][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs1[1]);
  MPI_Irecv(&x[s[0] - 1][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs1[2]);
  MPI_Irecv(&x[e[0] + 1][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs1[3]);

  MPI_Waitall(4, reqs1, MPI_STATUSES_IGNORE);

  }


  MPI_Datatype row_type;
  MPI_Type_vector(e[0] - s[0] + 1, 1, maxn, MPI_DOUBLE, &row_type); // We have to skip maxn as this is actually the size of the grid, not nx + 2.
  MPI_Type_commit(&row_type);

  if(mycoords[1] % 2 == 0){
    MPI_Request reqs2[4];

    MPI_Isend(&x[s[0]][s[1]], 1, row_type, nbrdown, 2, comm, &reqs2[0]);
    MPI_Irecv(&x[s[0]][e[1] + 1], 1, row_type, nbrup, 2, comm, &reqs2[1]);
    MPI_Isend(&x[s[0]][e[1]], 1, row_type, nbrup, 3, comm, &reqs2[2]);
    MPI_Irecv(&x[s[0]][s[1] - 1], 1, row_type, nbrdown, 3, comm, &reqs2[3]);
 
    MPI_Waitall(4, reqs2, MPI_STATUSES_IGNORE);

  } else {
    MPI_Request reqs3[4];

    MPI_Irecv(&x[s[0]][e[1] + 1], 1, row_type, nbrup, 2, comm, &reqs3[0]);
    MPI_Isend(&x[s[0]][s[1]], 1, row_type, nbrdown, 2, comm, &reqs3[1]);
    MPI_Irecv(&x[s[0]][s[1] - 1], 1, row_type, nbrdown, 3, comm, &reqs3[2]);
    MPI_Isend(&x[s[0]][e[1]], 1, row_type, nbrup, 3, comm, &reqs3[3]);

    MPI_Waitall(4, reqs3, MPI_STATUSES_IGNORE);

  }
}



void exchang3_2d(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown)
{


  MPI_Sendrecv(&x[e[0]][s[1]], e[1] - s[1] + 1 , MPI_DOUBLE, nbrright, 0, &x[s[0]-1][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrleft,
	       0, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&x[s[0]][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrleft, 1, &x[e[0]+1][s[1]], e[1] - s[1] + 1, MPI_DOUBLE, nbrright,
	       1, comm, MPI_STATUS_IGNORE);

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
			 int s[2], int e[2], double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright, int nbrup, int nbrdown)
{

  MPI_Request reqs[8];
  MPI_Status status;
  int idx;
  double h;
  int i,j,k;


  MPI_Datatype row_type;
  MPI_Type_vector(e[0] - s[0] + 1, 1, maxn, MPI_DOUBLE, &row_type); // We have to skip maxn as this is actually the size of the grid, not nx + 2.
  MPI_Type_commit(&row_type);

  int myid;
  MPI_Comm_rank(comm, &myid);

  //this assumes that the grid is a square
  h = 1.0/( (double)(nx+1) );

  MPI_Irecv(&u[s[0] - 1][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[0]);
  MPI_Irecv(&u[e[0] + 1][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[1]);
  MPI_Isend(&u[e[0]][s[1]], nx, MPI_DOUBLE, nbrright, 0, comm, &reqs[2]);
  MPI_Isend(&u[s[0]][s[1]], nx, MPI_DOUBLE, nbrleft, 0, comm, &reqs[3]);

  MPI_Irecv(&u[s[0]][s[1] - 1], 1, row_type, nbrdown, 0, comm, &reqs[4]);
  MPI_Irecv(&u[s[0]][e[1] + 1], 1, row_type, nbrup, 0, comm, &reqs[5]);
  MPI_Isend(&u[s[0]][s[1]], 1, row_type, nbrdown, 0, comm, &reqs[6]);
  MPI_Isend(&u[s[0]][e[1]], 1, row_type, nbrup, 0, comm, &reqs[7]);

  /* perform purely local updates (that don't need ghosts) */
  /* 2 cols or less means all are on processor boundary */
  if ((e[0] - s[0] + 1 >= 3) && (e[1] - s[1] + 1 >= 3)) {
    for (i = s[0] + 1; i <= e[0]; i++) {
      for (j = s[1] + 1; j < e[1]; j++) {
	unew[i][j] = 0.25 * ( u[i-1][j] + u[i+1][j] + u[i][j+1] + u[i][j-1]  - h*h*f[i][j] );
      }
    }
  }

 //We now preform the boundary conditions provided the opposite side of the boundary doesnt touch a ghost column/row. This requires e[0] -s[0] + 1 to be greater than two.
//This is the left boundary.
  if ((s[0] == 1) && (e[0] - s[0] + 1 >= 2)) {//We don't update the corners now. Wait for all communication to be done before I do this.
		for (j = s[1] + 1; j < e[1]; j++) {
			unew[1][j] = 0.25 * ( u[0][j] + u[2][j] + u[1][j+1] + u[1][j-1]  - h*h*f[1][j] );
		}
  }
//This is the right boundary.
  if ((e[0] == nx) && (e[0] - s[0] + 1 >= 2)) {//Again we don't update the corners.
		for (j = s[1] + 1; j < e[1]; j++) {
			unew[nx][j] = 0.25 * ( u[nx - 1][j] + u[nx +1][j] + u[nx][j+1] + u[nx][j-1]  - h*h*f[nx][j] );
		}
  }

  //Now update the bottom boundary provided it doesnt touch a ghost column.
  if ((s[1] == 1) && (e[1] - s[1] +1 >= 2)) {
	for (i = s[0] + 1; i < e[0]; i++) {
		unew[i][1] = 0.25 * ( u[i-1][1] + u[i+1][1] + u[i][2] + u[i][0] - h*h*f[i][0] );
	}
  }

  //Now update the top boundary provided it doesn't touch a ghost column.
  if ((e[1] == nx) && (e[1] - s[1] + 1 >= 2)) {
	for (i = s[0] + 1; i < e[0]; i++) {
		unew[i][nx] = 0.25 * ( u[i-1][nx] + u[i+1][nx] + u[i][nx + 1] + u[i][nx - 1]  - h*h*f[i][nx] );
	}
  }

  //MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
 for(k=0; k < 8; k++){
    MPI_Waitany(8, reqs, &idx, &status);

    // idx 0 is left recv, idx 1 is right recv, idx 4 is down recv, idx 5 is up recv 
    switch(idx){
      case 0:

      // left ghost update completed; update local leftmost column
        if (nbrleft != MPI_PROC_NULL && (e[0] - s[0] + 1 > 1)) { //The first condition is so we don't update the left boundary column again.
								 // The second condition ensures the update doesn't require the right process in addition.
          for(j = s[1] + 1; j < e[1]; j++){// We don't update the corners as this could require the top and bottom ghosts to have been completed.
	    unew[s[0]][j] = 0.25 * (u[s[0]-1][j] + u[s[0] + 1][j] + u[s[0]][j+1] + u[s[0]][j - 1]  - h*h*f[s[0]][j]);
          }
        }
        break;

      case 1:

      // left ghost update completed; update local leftmost column
        if (nbrright != MPI_PROC_NULL && (e[0] - s[0] + 1 > 1)) { //The first condition is so we don't update the right boundary column again.
								 // The second condition ensures the update doesn't require the left process in addition.
          for(j = s[1] + 1; j < e[1]; j++){// We don't update the corners as this could require the top and bottom ghosts to have been completed.
	    unew[e[0]][j] = 0.25 * (u[e[0]-1][j] + u[e[0] + 1][j] + u[e[0]][j+1] + u[e[0]][j - 1] - h*h*f[e[0]][j]);
		//printf("unew[e[0]][j] += 0.25 * (u[e[0]-1][j] + u[e[0] + 1][j] + u[e[0]][j+1] + u[e[0]][j - 1] - h*h*f[e[0]][j])\n", unew[e[0]][j] += 0.25 * (u[e[0]-1][j] + u[e[0] + 1][j] + u[e[0]][j+1] + u[e[0]][j - 1] - h*h*f[e[0]][j]);
          }
        }
        break;


      case 4:

      // left ghost update completed; update local leftmost column
        if (nbrdown != MPI_PROC_NULL && (e[1] - s[1] + 1 > 1)) { //The first condition is so we don't update the bottom boundary column again.
								 // The second condition ensures the update doesn't require the top process in addition.
          for(i = s[0] + 1; i < e[0]; i++){// We don't update the corners as this could require the top and bottom ghosts to have been completed.
	    unew[i][s[1]] = 0.25 * (u[i - 1][s[1]] + u[i + 1][s[1]] + u[i][s[1] + 1] + u[i][s[1] - 1] - h*h*f[i][s[1]]);
          }
        }
        break;


      case 5:

      // left ghost update completed; update local leftmost column
        if (nbrup != MPI_PROC_NULL && (e[1] - s[1] + 1 > 1)) { //The first condition is so we don't update the top boundary column again.
								 // The second condition ensures the update doesn't require the bottom process in addition.
          for(i = s[0] + 1; i < e[0]; i++){// We don't update the corners as this could require the top and bottom ghosts to have been completed.
	    unew[i][e[1]] = 0.25 * (u[i - 1][e[1]] + u[i + 1][e[1]] + u[i][e[1] + 1] + u[i][e[1] - 1] - h*h*f[i][e[1]]);
          }
        }
        break;

      default:
         break;
      }
   }

   //We now need to update the corners. This could be done in the swtich statement but that would create a large amount of if statements. 
   unew[s[0]][s[1]] = 0.25 * (u[s[0] - 1][s[1]] + u[s[0] + 1][s[1]] + u[s[0]][s[1] - 1] + u[s[0]][s[1] + 1] - h * h * f[s[0]][s[1]]);
   unew[s[0]][e[1]] = 0.25 * (u[s[0] - 1][e[1]] + u[s[0] + 1][e[1]] + u[s[0]][e[1] - 1] + u[s[0]][e[1] + 1] - h * h * f[s[0]][e[1]]);
   unew[e[0]][s[1]] = 0.25 * (u[e[0] - 1][s[1]] + u[e[0] + 1][s[1]] + u[e[0]][s[1] - 1] + u[e[0]][s[1] + 1] - h * h * f[e[0]][s[1]]);
   unew[e[0]][e[1]] = 0.25 * (u[e[0] - 1][e[1]] + u[e[0] + 1][e[1]] + u[e[0]][e[1] - 1] + u[e[0]][e[1] + 1] - h * h * f[e[0]][e[1]]);

  if ( e[1] - s[1] + 1 == 1) {
	for (i = s[0] + 1; i < e[0]; i++) {
	    unew[i][s[1]] = 0.25 * (u[i - 1][s[1]] + u[i + 1][s[1]] + u[i][s[1] + 1] + u[i][s[1] - 1] - h*h*f[i][s[1]]);
	}
  }

  if (e[0] -s[0] + 1 == 1) {
	for (j = s[1] + 1; j < e[1]; j++) {
	    unew[s[0]][j] = 0.25 * (u[s[0]-1][j] + u[s[0] + 1][j] + u[s[0]][j+1] + u[s[0]][j - 1]  - h*h*f[s[0]][j]);
	}
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
