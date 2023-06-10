#ifndef KERNEL_H
#define KERNEL_H
#include <fstream>
#include <mesh.h>
#include <mpi.h>
#include <omp.h>
#define __MPI

class kernel
{
    public:
    double *v;
    int nx, ny, nz;
    kernel(const char *filename);
    kernel &operator=(const kernel &k);
    ~kernel();
    double &operator()(int i, int j, int k);
};

double *integral_matrix(int narray, mesh *m, kernel *k);

#endif