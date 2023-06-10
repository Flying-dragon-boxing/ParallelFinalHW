#include "kernel.h"
#include "mesh.h"
#include "timer.h"

#ifdef __MPI
#include <mpi.h>
#endif

#define __OMP

#include <cstring>
#include <fstream>

venergy::venergy(const char *filename)
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
    else
    {
        nx = 0;
        ny = 0;
        nz = 0;
        v = nullptr;
    }
    
}

venergy::venergy()
{
    nx = 0;
    ny = 0;
    nz = 0;
    v = nullptr;
}

venergy &venergy::operator=(const venergy &k)
{
    nx = k.nx;
    ny = k.ny;
    nz = k.nz;
    v = new double[nx * ny * nz];
    memcpy(v, k.v, nx * ny * nz * sizeof(double));
    return *this;
}

venergy::~venergy()
{
    delete[] v;
}

double &venergy::operator()(int i, int j, int k)
{
    return v[i * ny * nz + j * nz + k];
}

#ifdef __MPI
double *integral_matrix(int narray, mesh *m, venergy &k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = m->nx * m->ny * m->nz;
    double *result = new double[narray*narray];
    memset(result, 0, narray * narray * sizeof(double));
    // manage task
    int *work = new int[narray*narray];
    int cnt = 0;
    for (int i = 0; i < narray; i++)
    {
        for (int j = 0; j < narray; j++)
        {
            if (i <= j)
            {                
                work[cnt++] = i * narray + j;
            }
        }
        
    }
    

    int nwork = cnt;
    int *nwork_per_process = new int[size];
    int *displs_work = new int[size];
    for (int i = 0; i < size; i++)
    {
        nwork_per_process[i] = nwork / size;
        if (i < nwork % size)
        {
            nwork_per_process[i]++;
        }
        displs_work[i] = i == 0 ? 0 : displs_work[i - 1] + nwork_per_process[i - 1];
    }
    
    int *gather_size = new int[size];
    int *displs_gather = new int[size];
    for (int i = 0; i < size; i++)
    {
        displs_gather[i] = i == 0 ? 0 : displs_gather[i - 1] + gather_size[i - 1];
        gather_size[i] = i == 0 ? work[nwork_per_process[i] - 1] +1: work[displs_work[i] + nwork_per_process[i] - 1] + 1 - displs_gather[i];
        
    }
    
    // calculate
    for (int i = 0; i < nwork_per_process[rank]; i++)
    {
        int index = work[displs_work[rank] + i];
        int i1 = index / narray;
        int i2 = index % narray;
        double sum = 0;
        #pragma omp parallel for reduction(+:sum) collapse(3)
        for (int x = 0; x < m->nx; x++)
        {
            for (int y = 0; y < m->ny; y++)
            {
                for (int z = 0; z < m->nz; z++)
                {
                    sum += k(x, y, z) * m[i2](x, y, z) * m[i1](x, y, z);
                }
                
            }
        }
        result[index] = sum * m->lx * m->ly * m->lz / (m->nx * m->ny * m->nz);
    }
    

    // gather at 0
    MPI_Gatherv(result + displs_gather[rank], gather_size[rank], MPI_DOUBLE, result, gather_size, displs_gather, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}

void venergy::mpi_bcast(int root)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Bcast(&nx, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank != root)
    {
        if (v != nullptr)
        {
            delete[] v;
        }
        v = new double[nx * ny * nz];
    }
    
    MPI_Bcast(v, nx * ny * nz, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

#endif