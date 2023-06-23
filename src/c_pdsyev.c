#include "c_pdsyev.h"
void c_pdsyev(int n, double *input_mat, double *eigenvalues, double *eigenvectors)
{
    // currently the matrix is read by all the MPI processes, this is inefficient
    int myrank_mpi, nprocs_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    MPI_Barrier(MPI_COMM_WORLD);
    int nprow, npcol;
    nprow = sqrt(nprocs_mpi);
    npcol = nprocs_mpi / nprow;
    int new_size = npcol * nprow;
    if (new_size != nprocs_mpi)
    {
        nprow = nprocs_mpi;
        npcol = 1;
        new_size = nprocs_mpi;
    }

    double *A = NULL;
    double *Z = NULL;
    double *W = NULL;

    int nb_row = n / nprow;
    int nb_col = n / npcol;
    if (myrank_mpi < n % nprow)
    {
        nb_row++;
    }
    if (myrank_mpi < n % npcol)
    {
        nb_col++;
    }
    nb_col = n;
    nb_row = n;

    

    // Initialize BLACS
    int iam, nprocs;
    int zero = 0;
    int ictxt, myrow, mycol;
    char layout = 'R';
    int izero = 0;

    blacs_pinfo_(&iam, &nprocs);                             // BLACS rank and world size
    blacs_get_(&zero, &zero, &ictxt);                        // -> Create context
    blacs_gridinit_(&ictxt, &layout, &nprow, &npcol);        // Context -> Initialize the grid
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol); // Context -> Context grid info (# procs row/col, current procs row/col)

    // Compute the size of the local matrices
    int mpA = numroc_(&n, &nb_row, &myrow, &izero, &nprow); // My proc -> row of local A
    int nqA = numroc_(&n, &nb_col, &mycol, &izero, &npcol); // My proc -> col of local A

    // printf("Proc %d/%d for MPI, proc %d/%d for BLACS in position (%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size %d\n",myrank_mpi,nprocs_mpi,iam,nprocs,myrow,mycol,nprow,npcol,mpA,nqA,n,nb);

    A = (double *)calloc(mpA * nqA, sizeof(double));
    if (A == NULL)
    {
        printf("error of memory allocation A on proc %dx%d\n", myrow, mycol);
        exit(0);
    }

    Z = (double *)calloc(mpA * nqA, sizeof(double));
    if (Z == NULL)
    {
        printf("error of memory allocation VT on proc %dx%d\n", myrow, mycol);
        exit(0);
    }

    int min_mn = n;
    W = (double *)calloc(min_mn, sizeof(double));
    if (W == NULL)
    {
        printf("error of memory allocation S on proc %dx%d\n", myrow, mycol);
        exit(0);
    }

    int k = 0;
    // printf("Proc %d/%d for MPI, proc %d/%d for BLACS at %d\n", myrank_mpi,nprocs_mpi,iam,nprocs, nb*(nb*myrow*npcol + mycol));

    // Create descriptor
    int descA[9];
    int descZ[9];
    int info;
    int lddA = mpA > 1 ? mpA : 1;
    descinit_(descA, &n, &n, &nb_row, &nb_col, &izero, &izero, &ictxt, &lddA, &info);
    descinit_(descZ, &n, &n, &nb_row, &nb_col, &izero, &izero, &ictxt, &lddA, &info);
    if (info != 0)
    {
        printf("Error in descinit, info = %d\n", info);
    }

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            // Read row major matrix
            pdelset_(A, &i, &j, descA, &(input_mat[(j - 1) + (i - 1) * n]));
        }
    }

    double *work = (double *)calloc(2, sizeof(double));
    if (work == NULL)
    {
        printf("error of memory allocation for work on proc %dx%d (1st time)\n", myrow, mycol);
        exit(0);
    }
    int lwork = -1;

    char jobz = 'V';
    char uplo = 'U';
    int ione = 1;

    pdsyev_(&jobz, &uplo, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info);

    lwork = (int)work[0];
    free(work);
    work = (double *)calloc(lwork, sizeof(double));
    if (work == NULL)
    {
        printf("error of memory allocation work on proc %dx%d\n", myrow, mycol);
        exit(0);
    }

    double MPIt1 = MPI_Wtime();
    pdsyev_(&jobz, &uplo, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info);

    double MPIt2 = MPI_Wtime();
    double MPIelapsed = MPIt2 - MPIt1;
    


    MPI_Barrier(MPI_COMM_WORLD);
    double *global_Z = (double*)calloc(n*n, sizeof(double));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            // Read row major matrix
            pdelget_("A", "I", &global_Z[(j - 1) + (i - 1) * n], Z, &i, &j, descZ);
        }
    }
    
    memcpy(eigenvalues, W, n * sizeof(double));
    memcpy(eigenvectors, global_Z, n * n * sizeof(double));
    // free(work);
    free(W);
    free(Z);
    free(A);


    // Exit and finalize
    blacs_gridexit_(&ictxt);

    return;
}
