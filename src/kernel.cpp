#include "kernel.h"
#include "mesh.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include <cstring>
#include <fstream>

kernel::kernel(const char *filename)
{
    if (filename != nullptr)
    {
        std::ifstream fin(filename);
        char buffer[100];
        fin >> buffer >> nx >> buffer >> ny >> buffer >> nz >> buffer;
        v = new double[nx * ny * nz];
        for (int i = 0; i < nx * ny * nz; i++)
        {
            fin >> v[i];
        }
    }
    
}

double &kernel::operator()(int i, int j, int k)
{
    return v[i * ny * nx + j * ny + k];
}

double integral(int nx, int ny, int nz, double *f1, double *f2, double *v)
{
    int n = nx * ny * nz;
    double result = 0;
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++)
    {
        result += f1[i] * f2[i] * v[i];
    }
    return result / n;
}

#ifdef __MPI
double *integral_matrix(int narray, mesh *m, kernel *k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nx , ny, nz,n;
    if (rank == 0)
    {
        nx = m->nx;
        ny = m->ny;
        nz = m->nz;
        n = nx * ny * nz;
    }
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&narray, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double *result = new double[narray * narray];
    double *f1 = new double[n*narray];
    double *v = new double[n];
    if (rank == 0)
    {
        for (int i = 0; i < narray; i++)
        {
            memcpy(f1 + i * n, (m+i)->m, n * sizeof(double));
        }
        memcpy(v, k->v, n * sizeof(double));
    }
    // scatterv by narray
    int *sendcounts = new int[size];
    int *displs = new int[size];
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = n / size;
        displs[i] = sum;
        sum += sendcounts[i];
    }
    sendcounts[size - 1] += n % size;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] *= narray;
        displs[i] *= narray;
    }
    MPI_Scatterv(f1, sendcounts, displs, MPI_DOUBLE, f1, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // calculate
    #pragma omp parallel for
    for (int cnt = displs[rank] / narray; cnt < (displs[rank] + sendcounts[rank]) / narray; cnt++)
    {
        int i = cnt / narray;
        int j = cnt % narray;
        if (i <= j)
        {
            result[i * narray + j] = integral(nx, ny, nz, f1 + i * n, f1 + j * n, v);
            // result[j * narray + i] = result[i * narray + j];
        }
        
    }
    // gatherv
    MPI_Gatherv(result, sendcounts[rank], MPI_DOUBLE, result, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}
#endif