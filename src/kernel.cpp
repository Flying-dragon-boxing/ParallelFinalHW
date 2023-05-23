#include "kernel.h"
#include "mesh.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include <fstream>

kernel::kernel(const char *filename)
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

double &kernel::operator()(int i, int j, int k)
{
    return v[i * ny * nx + j * ny + k];
}

#ifdef __MPI
double integral(kernel &v, mesh &f1, mesh &f2)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int n = f1.nx * f1.ny * f1.nz;
    double *result_local = new double[n];
    // scatter f1
    int *sendcounts = new int[size];
    int *displs = new int[size];
    int r = n % size;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = n / size;
        if (i < r)
        {
            sendcounts[i]++;
        }
        displs[i] = 0;
        for (int j = 0; j < i; j++)
        {
            displs[i] += sendcounts[j];
        }
    }
    double *f1_local = new double[sendcounts[rank]];
    MPI_Scatterv(f1.m, sendcounts, displs, MPI_DOUBLE, f1_local, sendcounts[rank], MPI_DOUBLE, 0, comm);
    // scatter f2
    double *f2_local = new double[sendcounts[rank]];
    MPI_Scatterv(f2.m, sendcounts, displs, MPI_DOUBLE, f2_local, sendcounts[rank], MPI_DOUBLE, 0, comm);
    // scatter v
    double *v_local = new double[sendcounts[rank]];
    MPI_Scatterv(v.v, sendcounts, displs, MPI_DOUBLE, v_local, sendcounts[rank], MPI_DOUBLE, 0, comm);
    // compute
    #pragma omp parallel for
    for (int i = 0; i < sendcounts[rank]; i++)
    {
        result_local[i] = f1_local[i] * f2_local[i] * v_local[i];
    }
    // reduce
    double result = 0;
    MPI_Reduce(&result, &result_local, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    return result / n * f1.lx * f1.ly * f1.lz;
}
#else
double integral(kernel &v, mesh &f1, mesh &f2)
{
    int n = f1.nx * f1.ny * f1.nz;
    double result = 0;
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++)
    {
        result += f1.m[i] * f2.m[i] * v.v[i];
    }
    return result / n * f1.lx * f1.ly * f1.lz;
}
#endif