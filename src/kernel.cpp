#include "kernel.h"
#include "mesh.h"
#include "timer.h"

#ifdef __MPI
#include <mpi.h>
#endif

#include <cstring>
#include <cassert>
#include <fstream>

venergy::venergy(const char *filename)
{
    if (filename != nullptr)
    {
        std::ifstream fin(filename);
        assert(fin.is_open());
        
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

void venergy::init(const char *filename)
{
    if (filename != nullptr)
    {
        std::ifstream fin(filename);
        assert(fin.is_open());
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

double &venergy::operator()(unsigned long long i, int j, int k)
{
    return v[i * ny * nz + j * nz + k];
}

#ifdef __MPI
double *integral_matrix(int narray, mesh *m, venergy &k, MPI_Comm comm, bool use_cache)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int nx = m->nx, ny = m->ny, nz = m->nz;
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
    
    double *cache = nullptr;
    int cache_pos = -1;
    // calculate
    double x_begin, x_end;
    double y_begin, y_end;
    double z_begin, z_end;
    for (int i = 0; i < nwork_per_process[rank]; i++)
    {
        int index = work[displs_work[rank] + i];
        int i1 = index / narray;
        int i2 = index % narray;
        double sum = 0;
        if (use_cache)
        {
            if (cache_pos == i1)
            {
#ifdef __OMP
                #pragma omp parallel for reduction(+:sum) collapse(3)
#endif           
                for (int x = x_begin; x < x_end; x++)
                {
                    for (int z = z_begin; z < z_end; z++)
                    {
                        for (int y = y_begin; y < y_end; y++)
                        {
                            sum += k(x, y, z) * m[i2](x, y, z) * cache[x * ny * nz + y * nz + z];
                        }
                        
                    }
                }
                result[index] = sum * m->lx * m->ly * m->lz / n;
            }
            else
            {
                x_begin = 0;
                if (m[i1].d.cutoff * nx / m->lx < m[i1].x)
                {
                    x_begin = m[i1].x - m[i1].d.cutoff * nx / m->lx;
                }
                x_end = nx;
                if (x_end > m->x + m[i1].d.cutoff * nx / m->lx)
                {
                    x_end = m->x + m[i1].d.cutoff * nx / m->lx;
                }
                y_begin = 0;
                if (m[i1].d.cutoff * ny / m->ly < m[i1].y)
                {
                    y_begin = m[i1].y - m[i1].d.cutoff * ny / m->ly;
                }
                y_end = ny;
                if (y_end > m->y + m[i1].d.cutoff * ny / m->ly)
                {
                    y_end = m->y + m[i1].d.cutoff * ny / m->ly;
                }
                z_begin = 0;
                if (m[i1].d.cutoff * nz / m->lz < m[i1].z)
                {
                    z_begin = m[i1].z - m[i1].d.cutoff * nz / m->lz;
                }
                z_end = nz;
                if (z_end > m->z + m[i1].d.cutoff * nz / m->lz)
                {
                    z_end = m->z + m[i1].d.cutoff * nz / m->lz;
                }
                
                if (cache != nullptr)
                {
                    delete[] cache;
                }
                cache_pos = i1;
                cache = new double[n];
#ifdef __OMP
                #pragma omp parallel for reduction(+:sum) collapse(3)
#endif
                for (int x = x_begin; x < x_end; x++)
                {
                    for (int z = z_begin; z < z_end; z++)
                    {
                        for (int y = y_begin; y < y_end; y++)
                        {
                            cache[x * ny * nz + y * nz + z] = m[i1](x, y, z);
                            sum += k(x, y, z) * m[i2](x, y, z) * cache[x * ny * nz + y * nz + z];
                        }
                        
                    }
                }
                result[index] = sum * m->lx * m->ly * m->lz / n;
            }
            
            
            
        }
        else
        {
            int x_begin = 0;
            if (m[i1].d.cutoff * nx / m->lx < m[i1].x)
            {
                x_begin = m[i1].x - m[i1].d.cutoff * nx / m->lx;
            }
            int x_end = nx;
            if (x_end > m->x + m[i1].d.cutoff * nx / m->lx)
            {
                x_end = m->x + m[i1].d.cutoff * nx / m->lx;
            }
            int y_begin = 0;
            if (m[i1].d.cutoff * ny / m->ly < m[i1].y)
            {
                y_begin = m[i1].y - m[i1].d.cutoff * ny / m->ly;
            }
            int y_end = ny;
            if (y_end > m->y + m[i1].d.cutoff * ny / m->ly)
            {
                y_end = m->y + m[i1].d.cutoff * ny / m->ly;
            }
            int z_begin = 0;
            if (m[i1].d.cutoff * nz / m->lz < m[i1].z)
            {
                z_begin = m[i1].z - m[i1].d.cutoff * nz / m->lz;
            }
            int z_end = nz;
            if (z_end > m->z + m[i1].d.cutoff * nz / m->lz)
            {
                z_end = m->z + m[i1].d.cutoff * nz / m->lz;
            }
                
#ifdef __OMP
            #pragma omp parallel for reduction(+:sum) collapse(3)
#endif
            for (int x = x_begin; x < x_end; x++)
            {
                for (int z = z_begin; z < z_end; z++)
                {
                    for (int y = y_begin; y < y_end; y++)
                    {
                        sum += k(x, y, z) * m[i2](x, y, z) * m[i1](x, y, z);
                    }
                    
                }
            }
            result[index] = sum * m->lx * m->ly * m->lz / n;
        }
        
    }
    
    delete[] cache;
    // gather at 0
    // MPI_Gatherv(result + displs_gather[rank], gather_size[rank], MPI_DOUBLE, result, gather_size, displs_gather, MPI_DOUBLE, 0, comm);
    MPI_Allgatherv(result + displs_gather[rank], gather_size[rank], MPI_DOUBLE, result, gather_size, displs_gather, MPI_DOUBLE, comm);
    return result;
}

void venergy::mpi_bcast(MPI_Comm comm, int root)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Bcast(&nx, 1, MPI_INT, root, comm);
    MPI_Bcast(&ny, 1, MPI_INT, root, comm);
    MPI_Bcast(&nz, 1, MPI_INT, root, comm);
    if (rank != root)
    {
        if (v != nullptr)
        {
            delete[] v;
        }
        v = new double[nx * ny * nz];
    }
    
    MPI_Bcast(v, nx * ny * nz, MPI_DOUBLE, root, comm);
}

#endif