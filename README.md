# MPI-numerical-solution-to-the-poisson-equation

The code along with a makefile can be found within the Tarball file. The codes submitted in the
assignment are assignment2_part_1_to_3.c, which contains the code for parts 1 to 3, blocking_2d_decomp.c,
which the 2d blocking decomposition and nonblocking_2d_decomp.c, which is the nonblocking 2d decomposition.
These codes call jacobi.c, which contains the exchange and sweeping functions for 1d and 2d while decomp1d.c
which contains the 1d decompositon function.

On chuck, the MPI modules must be loaded with the command 'module load cports openmpi'

Run the makefile with 'make'

This will generate the execuatables assignment2-part_1to_3, blocking_2d_decomp,
and nonblocking_2d-decomp which are the execuatbles for the corresponding c files.

Each executable can be run with 'mpirun -np <number of processors> <executable name> <size of grid>

This will print the analytic solution for that grid size and the numerical grid which has been
gather onto the rank zero process and print from there

textfiles called 'grid for processor <rank>' will also be generated for assignment2_part_1to_3
which contains the local grid that each processor has.

