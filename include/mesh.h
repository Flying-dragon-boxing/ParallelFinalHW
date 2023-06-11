#ifndef MESH_H
#define MESH_H

#include <mpi.h>

double spline(double x, double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3);

class dist
{
    public:
    double dx, cutoff;
    double *m;
    int n;
    dist(const char *filename);
    dist();
    dist &operator=(const dist &d);
    ~dist();
    double operator()(double x);
    void mpi_bcast(MPI_Comm comm, int root = 0);
};

class mesh
{
    public:
    // static dist &d;
    int nx, ny, nz;
    double lx, ly, lz;
    double x, y, z;
    dist d;
    mesh(int _x, int _y, int _z, double lx, double ly, double lz);
    mesh();
    mesh(const mesh &m);
    mesh &operator=(const mesh &m);
    void init(double x, double y, double z, dist &d);
    double operator()(unsigned long long i, int j, int k);
    void mpi_bcast(MPI_Comm comm, int root = 0);
};

#endif