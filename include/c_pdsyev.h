#ifndef C_PDSYEV_H
#define C_PDSYEV_H
#ifdef __cplusplus
extern "C"
{
    #include <mpi.h>
    #include <math.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>

    void blacs_get_(int *, int *, int *);
    void blacs_pinfo_(int *, int *);
    void blacs_gridinit_(int *, char *, int *, int *);
    void blacs_gridinfo_(int *, int *, int *, int *, int *);
    void descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
    void blacs_gridexit_(int *);
    int numroc_(int *, int *, int *, int *, int *);
    void pdsyev_(char *jobz, char *uplo, int *n, double *a, int *ia, int *ja,
                int *desca, double *w, double *z, int *iz, int *jz, int *descz,
                double *work, int *lwork, int *info);
    void pdelset_(double *mat, const int *i, const int *j, const int *desc, const double *a);
    void pdelget_(char *scope, char *top, double *alpha, double *mat, const int *i, const int *j, const int *desc);
    void c_pdsyev(int n, double *input_mat, double *eigenvalues, double *eigenvectors);
}
#else
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void blacs_get_(int *, int *, int *);
void blacs_pinfo_(int *, int *);
void blacs_gridinit_(int *, char *, int *, int *);
void blacs_gridinfo_(int *, int *, int *, int *, int *);
void descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
void blacs_gridexit_(int *);
int numroc_(int *, int *, int *, int *, int *);
void pdsyev_(char *jobz, char *uplo, int *n, double *a, int *ia, int *ja,
            int *desca, double *w, double *z, int *iz, int *jz, int *descz,
            double *work, int *lwork, int *info);
void pdelset_(double *mat, const int *i, const int *j, const int *desc, const double *a);
void pdelget_(char *scope, char *top, double *alpha, double *mat, const int *i, const int *j, const int *desc);
void c_pdsyev(int n, double *input_mat, double *eigenvalues, double *eigenvectors);
#endif
#endif