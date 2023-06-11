#include <iostream>
#include <fstream>
// #include <sstream>
#include <string>
#include <map>
#include <mpi.h>
#include <memory>
#include <unistd.h>

#include "kernel.h"
#include "mesh.h"
#include "timer.h"

#define __MPI
// #define __MPI_DEBUG

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::ios::sync_with_stdio(false);
    if (rank == 0)
        timer::tick("", "total");

    std::map<std::string, std::string> args;
    std::string filename = "../input/INPUT.txt";

    int n_points, nx, ny, nz;
    mesh *pm = nullptr;
    venergy k(nullptr);
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
        if (args.find("venergy_path") != args.end())
        {
            v_path = args["venergy_path"];
        }
        if (args.find("distribution_path") != args.end())
        {
            distribution_path = args["distribution_path"];
        }

        dist d(("../input/"+distribution_path).c_str());
        k = venergy(("../input/"+v_path).c_str());
        
        nx = k.nx;
        ny = k.ny;
        nz = k.nz;
        
        // read points
        std::ifstream fin_points(points_path.c_str());
        n_points = 0;
        char buffer[504];
        while (fin_points.getline(buffer, 500))
        {
            if (buffer[0] != ' ' && buffer[0] != '\n')
            {
                n_points ++;
            }
            
        }
        fin_points.clear();
        fin_points.seekg(0, std::ios::beg);
        pm = new mesh[n_points];
        double *points = new double[n_points * 3];
        for (int i = 0; i < n_points; i++)
        { 
#ifdef __INPUT_CURLY
            buffer[0] = fin_points.get();
            fin_points >> points[i * 3] >> buffer >> points[i * 3 + 1] >> buffer >> points[i * 3 + 2] >> buffer;
#else
            fin_points >> points[i * 3] >> points[i * 3 + 1] >> points[i * 3 + 2];
#endif
            fin_points.getline(buffer, 500);
        }
        for (int i = 0; i < n_points; i++)
        {
            pm[i] = mesh(nx, ny, nz, lx, ly, lz);
            pm[i].init(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], d);
        }
        
    }
    
#ifdef __MPI_DEBUG
    sleep(10);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        pm = new mesh[n_points];
    }
    
    MPI_Comm Brave_New_World;
    unsigned long long max_memory = (unsigned long long) 12 * 1024 * 1024 * 1024;
    int max_size = max_memory / (sizeof(double) * nx * ny * nz);
    int color = (rank < max_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &Brave_New_World);
    double *return_matrix = nullptr;
    if (rank < max_size)
    {
        std::string func_name = "integral_matrix"+std::to_string(rank);
        for (int i = 0; i < n_points; i++)
        {
            pm[i].mpi_bcast(Brave_New_World);
        }
        k.mpi_bcast(Brave_New_World);
        
        MPI_Barrier(Brave_New_World);
        std::cout << "rank " << rank << " start with venergy " << k.v[1] << std::endl;
        timer::tick("", func_name);
        return_matrix = integral_matrix(n_points, pm, k, Brave_New_World);
        // double *return_matrix = new double[n_points * n_points];
        timer::tick("", func_name);
        timer::mpi_sync();
    }
    
    if (rank == 0)
        timer::tick("", "total");
    if (rank == 0)
        timer::print();

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
    delete[] return_matrix;
    MPI_Finalize();
}