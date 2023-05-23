#ifndef KERNEL_H
#define KERNEL_H
#include <fstream>
#include <mpi.h>
#include <omp.h>
#define __MPI

class kernel
{
    public:
    double *v;
    int nx, ny, nz;
    kernel(const char *filename);
    double &operator()(int i, int j, int k);
};

double integral(kernel &v, mesh &f1, mesh &f2);

#endif