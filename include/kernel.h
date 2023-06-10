#ifndef KERNEL_H
#define KERNEL_H
#include <fstream>
#include <mesh.h>
#include <mpi.h>
#include <omp.h>
#define __MPI

class venergy
{
    public:
    double *v;
    int nx, ny, nz;
    venergy(const char *filename);
    venergy();
    venergy &operator=(const venergy &k);
    ~venergy();
    double &operator()(int i, int j, int k);
    void mpi_bcast(int root = 0);
};

double *integral_matrix(int narray, mesh *m, venergy &k);

#endif