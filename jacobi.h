void sweep1d(double a[][maxn], double f[][maxn], int nx,
	     int s, int e, double b[][maxn]);

void  sweep2d(double a[][maxn], double f[][maxn], int nx, int s[2], int e[2], double b[][maxn]);

void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright);

void exchang3_2d(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown);

void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s, int e, double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright);

void nbxchange_and_sweep_2d(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s[2], int e[2], double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright, int nbrup, int nbrdown);

double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e);

double griddiff_2d(double a[][maxn], double b[][maxn], int nx, int s[2], int e[2]);

void exchang3_2d_nb_sendrecv(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown, int mycoords[2]);

