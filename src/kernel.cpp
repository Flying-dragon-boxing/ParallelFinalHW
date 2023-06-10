#include "kernel.h"
#include "mesh.h"
#include "timer.h"

#ifdef __MPI
#include <mpi.h>
#endif

#define __OMP

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

kernel &kernel::operator=(const kernel &k)
{
    nx = k.nx;
    ny = k.ny;
    nz = k.nz;
    v = new double[nx * ny * nz];
    memcpy(v, k.v, nx * ny * nz * sizeof(double));
    return *this;
}

kernel::~kernel()
{
    delete[] v;
}

double &kernel::operator()(int i, int j, int k)
{
    return v[i * ny * nz + j * nz + k];
}

#ifdef __MPI
double *integral_matrix(int narray, mesh *m, kernel *k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nx, ny, nz, n;
    double lx, ly, lz, l;
    if (rank == 0)
    {
        nx = m->nx;
        ny = m->ny;
        nz = m->nz;
        lx = m->lx;
        ly = m->ly;
        lz = m->lz;
        l = lx*ly*lz;
        n = nx * ny * nz;
    }
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&narray, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
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
    
    // allocate task
    int *sendcounts = new int[size];
    int *displs = new int[size];
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = narray * narray / size;
        if (i < narray * narray % size)
        {
            sendcounts[i]++;
        }
        
        displs[i] = sum;
        sum += sendcounts[i];
    }

    timer::tick("","pre_integral"+std::to_string(rank));
    MPI_Bcast(v, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f1, n * narray, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    timer::tick("","pre_integral"+std::to_string(rank));
    // calculate
    
    // #pragma omp parallel for thread_num(4)
    for (int i = displs[rank]; i < (displs[rank] + sendcounts[rank]); i++)
    {
        int i1 = i / narray;
        int i2 = i % narray;
        if (i1 > i2)
        {
            continue;
        }    
        // #pragma omp atomic
        timer::tick("","integral"+std::to_string(rank));
        double resi = 0;
#ifdef __OMP
        #pragma omp parallel for reduction(+:resi) thread_num(4)
#endif
        for (int j = 0; j < n; j++)
        {

            resi += (f1 + i1 * n)[j] * (f1 + i2 * n)[j] * v[j];       
            
        }
        result[i] = resi;
        timer::tick("","integral"+std::to_string(rank));
    }
    // gatherv
    MPI_Gatherv(result + displs[rank], sendcounts[rank], MPI_DOUBLE, result, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] f1;
    delete[] v;
    return result;
}
#endif