#include "mesh.h"
#include <mpi.h>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cassert>

double spline(double x, double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3)
{
    double a0, a1, a2, a3;
    a0 = y0;
    a1 = (y1 - y0) / (x1 - x0);
    a2 = (y2 - 2 * y1 + y0) / ((x2 - x1) * (x2 - x0));
    a3 = (y3 - 3 * y2 + 3 * y1 - y0) / ((x3 - x2) * (x3 - x1) * (x3 - x0));
    return a0 + a1 * (x - x0) + a2 * (x - x0) * (x - x1) + a3 * (x - x0) * (x - x1) * (x - x2);
}

dist::dist(const char *filename)
{
    std::ifstream fin(filename);
    assert(fin.is_open());
    char buffer[100];
    fin >> buffer >> cutoff >> buffer >> dx >> buffer >> n >> buffer >> buffer >> buffer;
    m = new double[n];
    for (int i = 0; i < n; i++)
    {
        fin >> m[i] >> buffer;
    }
    
}

dist::dist()
{
    dx = 0;
    cutoff = 0;
    n = 0;
    m = nullptr;
}

dist &dist::operator=(const dist &d)
{
    dx = d.dx;
    cutoff = d.cutoff;
    n = d.n;
    if (m != nullptr)
    {
        delete[] m;
    }
    
    m = new double[n];
    memcpy(m, d.m, n * sizeof(double));
    return *this;
}

dist::~dist()
{
    delete[] m;
}

double dist::operator()(double x)
{
    if (x < 0)
        return 0;
    if (x > cutoff)
        return 0;
    int i = (int)(x / dx);
    if (i < 3)
        return spline(x, 0 * dx, 1 * dx, 2 * dx, 3 * dx, m[0], m[1], m[2], m[3]);
    return spline(x, (i - 3) * dx, (i - 2) * dx, (i - 1) * dx, i * dx, m[i - 3], m[i - 2], m[i - 1], m[i]);
}

void dist::mpi_bcast(MPI_Comm comm, int root)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Bcast(&dx, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&cutoff, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&n, 1, MPI_INT, root, comm);
    if (m == nullptr)
        m = new double[n];
    else if (root != rank)
    {
        delete[] m;
        m = new double[n];
    }
    
    MPI_Bcast(m, n, MPI_DOUBLE, root, comm);
}

mesh::mesh(int _x, int _y, int _z, double _lx, double _ly, double _lz)
{
    nx = _x;
    ny = _y;
    nz = _z;
    lx = _lx;
    ly = _ly;
    lz = _lz;
    x = 0;
    y = 0;
    z = 0;

}

mesh::mesh()
{
    nx = 0;
    ny = 0;
    nz = 0;
    lx = 0;
    ly = 0;
    lz = 0;
    x = 0;
    y = 0;
    z = 0;
}

mesh::mesh(const mesh &m)
{
    nx = m.nx;
    ny = m.ny;
    nz = m.nz;
    lx = m.lx;
    ly = m.ly;
    lz = m.lz;
    x = m.x;
    y = m.y;
    z = m.z;
    d = m.d;
}

mesh &mesh::operator=(const mesh &m)
{
    nx = m.nx;
    ny = m.ny;
    nz = m.nz;
    lx = m.lx;
    ly = m.ly;
    lz = m.lz;
    x = m.x;
    y = m.y;
    z = m.z;
    d = m.d;
    return *this;
}

void mesh::init(double _x, double _y, double _z, dist &_d)
{
    x = _x;
    y = _y;
    z = _z;
    d = _d;
}

double mesh::operator()(unsigned long long i, int j, int k)
{
    return d(std::sqrt((lx*i/nx - x) * (lx*i/nx - x) + (ly*j/ny - y) * (ly*j/ny - y) + (lz*k/nz - z) * (lz*k/nz - z)));
}

void mesh::mpi_bcast(MPI_Comm comm, int root)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Bcast(&nx, 1, MPI_INT, root, comm);
    MPI_Bcast(&ny, 1, MPI_INT, root, comm);
    MPI_Bcast(&nz, 1, MPI_INT, root, comm);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&x, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&y, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&z, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&d.dx, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&d.cutoff, 1, MPI_DOUBLE, root, comm);
    MPI_Bcast(&d.n, 1, MPI_INT, root, comm);
    if (nx * ny * nz > 0)
    {
        if (root != rank)
        {
            if (d.m != nullptr)
            {
                delete[] d.m;  
            }
            
            d.m = new double[d.n];
        }
        
        MPI_Bcast(d.m, d.n, MPI_DOUBLE, root, comm);
    }
}