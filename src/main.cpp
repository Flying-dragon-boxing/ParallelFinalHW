#include <iostream>
#include <fstream>
// #include <sstream>
#include <string>
#include <map>
#include <mpi.h>
#include <memory>

#include "kernel.h"
#include "mesh.h"

#define __MPI

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::map<std::string, std::string> args;
    std::string filename = "input.txt";

    int n_points;
    mesh *pm = nullptr;
    kernel k(nullptr);
    if (rank == 0)
    {
        std::ifstream fin(filename);
        std::string key, value;
        while (fin >> key >> value)
        {
            args[key] = value;
        }
        
        double lx, ly, lz;
        std::string diago_lib, points_path, v_path, distribution_path;
        if (args.find("lx") != args.end())
        {
            lx = std::stod(args["lx"]);
        }
        if (args.find("ly") != args.end())
        {
            ly = std::stod(args["ly"]);
        }
        if (args.find("lz") != args.end())
        {
            lz = std::stod(args["lz"]);
        }
        if (args.find("diago_lib") != args.end())
        {
            diago_lib = args["diago_lib"];
        }
        else
        {
            diago_lib = "lapack";
        }
        if (args.find("points_path") != args.end())
        {
            points_path = args["points_path"];
        }
        if (args.find("v_path") != args.end())
        {
            v_path = args["v_path"];
        }
        if (args.find("distribution_path") != args.end())
        {
            distribution_path = args["distribution_path"];
        }

         dist d(distribution_path.c_str());
        k = kernel(v_path.c_str());
        int nx, ny, nz;
        nx = k.nx;
        ny = k.ny;
        nz = k.nz;
        
        // read points
        std::ifstream fin_points(points_path.c_str());
        fin_points >> n_points;
        double *points = new double[n_points * 3];
        for (int i = 0; i < n_points; i++)
        {
            fin_points >> points[i * 3] >> points[i * 3 + 1] >> points[i * 3 + 2];
        }

        std::allocator<mesh> alloc_mesh;
        pm = alloc_mesh.allocate(n_points);
        for (int i = 0; i < n_points; i++)
        {
            alloc_mesh.construct(pm + i, nx, ny, nz, lx, ly, lz);
            (pm+i)->init(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], d);
        }
    }
    
    
    double *return_matrix = integral_matrix(n_points, pm, &k);
    if (rank == 0)
    {
        std::ofstream fout("result.txt");
        for (int i = 0; i < n_points; i++)
        {
            for (int j = 0; j < n_points; j++)
            {
                fout << return_matrix[i * n_points + j] << " ";
            }
            fout << std::endl;
        }
    }
    
}