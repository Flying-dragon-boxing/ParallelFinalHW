#ifndef MESH_H
#define MESH_H
double spline(double x, double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3);

class dist
{
    public:
    double dx, cutoff;
    double *m;
    int n;
    dist(const char *filename);
    double operator()(double x);
};

class mesh
{
    public:
    // static dist &d;
    int nx, ny, nz;
    double *m;
    double lx, ly, lz;
    mesh(int _x, int _y, int _z, double lx, double ly, double lz);
    void init(double x, double y, double z, dist &d);
    double &operator()(int i, int j, int k);
};

#endif