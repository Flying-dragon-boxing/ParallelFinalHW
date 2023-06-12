#include <iostream>
#include <fstream>
// #include <sstream>
#include <string>
#include <map>
#include <mpi.h>
#include <memory>
#include <unistd.h>
#ifdef __OMP
#include <omp.h>
#endif
#include <lapacke.h>
#include <cassert>

#include "kernel.h"
#include "mesh.h"
#include "timer.h"
#include "c_pdsyev.h"


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
    std::string diago_lib;

    int n_points, nx, ny, nz;
    mesh *pm = nullptr;
    venergy *k = new venergy(nullptr);
    char diago_lib_cstr[20] = "";
    if (rank == 0)
    {
        std::ifstream fin(filename);
        assert(fin.is_open());
        std::string key, value;
        while (fin >> key >> value)
        {
            args[key] = value;
        }
        
        double lx, ly, lz;
        std::string points_path, v_path, distribution_path;
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
        memcpy(diago_lib_cstr, diago_lib.c_str(), diago_lib.length());
        dist d((distribution_path).c_str());
        k->init((v_path).c_str());
        
        nx = k->nx;
        ny = k->ny;
        nz = k->nz;
        
        // read points
        std::ifstream fin_points(points_path.c_str());
        assert(fin_points.is_open());
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
    sleep(15);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(diago_lib_cstr, 20, MPI_CHAR, 0, MPI_COMM_WORLD);
    diago_lib = diago_lib_cstr;
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        pm = new mesh[n_points];
    }

    bool use_cache = false;
    MPI_Comm Brave_New_World;
    unsigned long long max_memory = (unsigned long long) 10 * 1024 * 1024 * 1024;
    int max_size = max_memory / sizeof(double) / nx / ny / nz;
    if (max_size > 4)
    {
        max_size /= 2;
        use_cache = true;
    }
#ifdef __OMP
    omp_set_num_threads(2);
    if (size / max_size > 1)
    {
        omp_set_num_threads(2 * (size / max_size + 1));
    }

    
#endif    
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
        k->mpi_bcast(Brave_New_World);
        
        MPI_Barrier(Brave_New_World);
        std::cout << "rank " << rank << " start with venergy " << k->v[1] << std::endl;
        timer::tick("", func_name);
        return_matrix = integral_matrix(n_points, pm, *k, Brave_New_World, use_cache);
        // double *return_matrix = new double[n_points * n_points];
        timer::tick("", func_name);
        timer::mpi_sync(Brave_New_World);
    }
    
    if (rank == 0)
        timer::tick("", "total");
    if (rank == 0)
        timer::print();


    delete k;

    if (rank == 0)
    {
        std::ofstream fout_result("result.txt");
        for (int i = 0; i < n_points; i++)
        {
            for (int j = 0; j < n_points; j++)
            {
                fout_result << return_matrix[i * n_points + j] << " ";
            }
            fout_result << std::endl;
        }
    }

    double *eigenvalue = nullptr;
    double *eigenvector = nullptr;
    // use lapacke at rank 0
    // if (diago_lib == "lapack" && rank == 0)
    // {

    //     eigenvalue = new double[n_points];
    //     eigenvector = new double[n_points * n_points];
    //     LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n_points, return_matrix, n_points, eigenvalue);
    //     std::ofstream fout("eigenvalue.txt");
    //     for (int i = 0; i < n_points; i++)
    //     {
    //         fout << eigenvalue[i] << std::endl;
    //     }
    //     fout.close();
    //     fout.open("eigenvector.txt");
    //     for (int i = 0; i < n_points; i++)
    //     {
    //         for (int j = 0; j < n_points; j++)
    //         {
    //             fout << return_matrix[i * n_points + j] << " ";
    //         }
    //         fout << std::endl;
    //     }

    
    // }
    // // or use scalapack
    // else if (diago_lib == "scalapack")
    {
        MPI_Bcast(return_matrix, n_points * n_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double *eigenvalue = new double[n_points];
        double *eigenvector = new double[n_points * n_points];
        c_pdsyev(n_points, return_matrix, eigenvalue, eigenvector);
        if (rank == 0)
        {
            std::ofstream fout("eigenvalue.txt");
            for (int i = 0; i < n_points; i++)
            {
                fout << eigenvalue[i] << std::endl;   
            }
            fout.close();
            fout.open("eigenvector.txt");
            for (int i = 0; i < n_points; i++)
            {
                for (int j = 0; j < n_points; j++)
                {
                    fout << eigenvector[i * n_points + j] << " ";
                }
                fout << std::endl;
            }
        }

        
        
    }

    
    delete[] return_matrix;
    MPI_Finalize();
}
