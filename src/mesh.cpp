#include "mesh.h"
#include <omp.h>
#include <cmath>
#include <fstream>
double spline(double x, double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3)
{
    double a0, a1, a2, a3;
    a0 = y0;
    a1 = (y1 - y0) / (x1 - x0);
    a2 = (y2 - 2 * y1 + y0) / ((x2 - x1) * (x2 - x0));
    a3 = (y3 - 3 * y2 + 3 * y1 - y0) / ((x3 - x2) * (x3 - x1) * (x3 - x0));
    return a0 + a1 * (x - x0) + a2 * (x - x0) * (x - x1) + a3 * (x - x0) * (x - x1) * (x - x2);
}

mesh::mesh(int _x, int _y, int _z, double _lx, double _ly, double _lz)
{
    nx = _x;
    ny = _y;
    nz = _z;
    lx = _lx;
    ly = _ly;
    lz = _lz;
    m = new double[nx * ny * nz];
}

void mesh::init(double x, double y, double z)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                double n_x = (i + 0.5) * lx / nx;
                double n_y = (j + 0.5) * ly / ny;
                double n_z = (k + 0.5) * lz / nz;
                double r = sqrt((n_x - x) * (n_x - x) + (n_y - y) * (n_y - y) + (n_z - z) * (n_z - z));
                (*this)(i, j, k) = d(r);
            }
            
        }
        
    }
    
}

double &mesh::operator()(int i, int j, int k)
{
    return m[i * ny * nz + j * nz + k];
}

dist::dist(const char *filename)
{
    std::ifstream fin(filename);
    char buffer[100];
    fin >> buffer >> cutoff >> buffer >> dx >> buffer >> n >> buffer >> buffer >> buffer;
    m = new double[n];
    for (int i = 0; i < n; i++)
    {
        fin >> m[i];
    }
    
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
