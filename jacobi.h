void sweep1d(double a[][maxn], double f[][maxn], int nx,
	     int s, int e, double b[][maxn]);

void  sweep2d(double a[][maxn], double f[][maxn], int nx, int s[2], int e[2], double b[][maxn]);

void exchang1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrbottom, int nbrtop);

void exchang2(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright);

void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright);

void exchang3_2d(double x[][maxn], int nx, int s[2], int e[2], MPI_Comm comm,
	      int nbrleft, int nbrright, int nbrup, int nbrdown);
 

void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	       int nbrleft, int nbrright);

void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s, int e, double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright);

double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e);

double griddiff_2d(double a[][maxn], double b[][maxn], int nx, int s[2], int e[2]);

