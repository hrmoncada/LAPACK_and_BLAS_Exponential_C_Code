/* expm_lapack_dsyev.c - Calculate the exponential of a symmetric rectangular matrix A
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

    DSYEVR : Computes selected eigenvalues and, optionally, eigenvectors   
    of a real symmetric matrix A.  Eigenvalues and eigenvectors can be   
    selected by specifying either a range of values or a range of   
    indices for the desired eigenvalues.   

    DSYEVR first reduces the matrix A to tridiagonal form T with a call   
    to DSYTRD.  Then, whenever possible, DSYEVR calls DSTEMR to compute   
    the eigenspectrum using Relatively Robust Representations.  DSTEMR   
    computes eigenvalues by the dqds algorithm, while orthogonal   
    eigenvectors are computed from various "good" L D L^T representations   
    (also known as Relatively Robust Representations). Gram-Schmidt   
    orthogonalization is avoided as far as possible. More specifically,   
    the various steps of the algorithm are as follows.   

    For each unreduced block (submatrix) of T,   
       (a) Compute T - sigma I  = L D L^T, so that L and D   
           define all the wanted eigenvalues to high relative accuracy.   
           This means that small relative changes in the entries of D and L   
           cause only small relative changes in the eigenvalues and   
           eigenvectors. The standard (unfactored) representation of the   
           tridiagonal matrix T does not have this property in general.   
       (b) Compute the eigenvalues to suitable accuracy.   
           If the eigenvectors are desired, the algorithm attains full   
           accuracy of the computed eigenvalues only right before   
           the corresponding vectors have to be computed, see steps c) and d).   
       (c) For each cluster of close eigenvalues, select a new   
           shift close to the cluster, find a new factorization, and refine   
           the shifted eigenvalues to suitable accuracy.   
       (d) For each eigenvalue with a large enough relative separation compute   
           the corresponding eigenvector by forming a rank revealing twisted   
           factorization. Go back to (c) for any clusters that remain.   

    The desired accuracy of the output can be specified by the input   
    parameter ABSTOL. 

Compile:
>> gcc expm_lapack_dsyevr.c -lblas -llapack -lm  -L/usr/lib -o out

>> gcc expm_lapack_dsyevr.c -lblas -llapack -lm   -o out

Execute:
>> ./out 

Solve Possible Error :  
   fatal error: cblas.h: No such file or directory compilation terminated. 
To solved the issue install:  
     $ sudo apt-get install libopenblas-dev 

 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

 static const int N = 2; // Matrix size
 
 static int  dsyevr(char JOBZ, char RANGE, char UPLO, int N, double *A, int LDA, double VL, double VU,
		    int IL, int IU, double ABSTOL, int *M, double *W, double *Z, int LDZ, int *ISUPPZ,
		    double *WORK, int LWORK, int *IWORK, int LIWORK) {
             
 extern void dsyevr_(char *JOBZp,char *RANGEp, char *UPLOp, int *Np, double *A, int *LDAp, double *VLp,
		     double *VUp, int *ILp, int *IUp, double *ABSTOLp, int *Mp, double *W, double *Z,
		     int *LDZp, int *ISUPPZ,double *WORK, int *LWORKp, int *IWORK, int *LIWORKp,
		     int *INFOp);
 
            int INFO;
       
            dsyevr_(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, M, W, Z, &LDZ,
		    ISUPPZ, WORK, &LWORK, IWORK, &LIWORK, &INFO);
	    
            return INFO;
 }

 static double dlamch(char CMACH) {
      extern double dlamch_(char *CMACHp);
      return dlamch_(&CMACH);
 }

 double **CREAT_MATRIX();     /*Function*/

 int main() {
  double *A, *B, *W, *Z, *WORK;
  int *ISUPPZ, *IWORK;
  int  i, j;
  int  M;
  /* allocate and initialise the matrix */
  A = malloc(N*N*sizeof(double));  
  
/* A must be SYMMETRIC SEMI-POSITIVE DEFINIT matrix*/  
  double **AA = CREAT_MATRIX (N , N);  //( rows,colum)
  AA[0][0] = 5.0;   AA[0][1] = 4.0;    
  AA[1][0] = 4.0;   AA[1][1] = 5.0; 
 
/*AA[0][0] =  1.96;   AA[0][1] = -6.49;  AA[0][2] = -0.47;   AA[0][3] = -7.20;   AA[0][4] = -0.65;
  AA[1][0] = -6.49;   AA[1][1] =  3.80;  AA[1][2] = -6.39;   AA[1][3] = 1.50;    AA[1][4] = -6.34;
  AA[2][0] = -0.47;   AA[2][1] = -6.39;  AA[2][2] =  4.17;   AA[2][3] = -1.51;   AA[2][4] =  2.67;
  AA[3][0] = -7.20;   AA[3][1] =  1.50;  AA[3][2] = -1.51;   AA[3][3] =  5.70;   AA[3][4] =  1.80;
  AA[4][0] = -0.65;   AA[4][1] = -6.34;  AA[4][2] =  2.67;   AA[4][3] =  1.80;   AA[4][4] = -7.10; */
    printf("\nMatrix :\n");
  for (i = 0; i < N; i++){  
      for (j = 0; j < N; j++) {       
         printf(" A[%d][%d]  =  %4.2f ",i ,j , AA[i][j]);
      }
      printf("\n");
  }
  
  for (j = 0;j < N; j++) { 
      for (i = 0;i < N; i++){           
	 A[i + N*j] = AA[i][j];
         //printf(" A[%d] = %2.1f\n",i + N*j, A[i + N*j]);
      }
  }

  /* allocate space for the output parameters and workspace arrays */
  W = malloc(N*sizeof(double));
  Z = malloc(N*N*sizeof(double));
  ISUPPZ = malloc(2*N*sizeof(int));
  WORK = malloc(26*N*sizeof(double));
  IWORK = malloc(10*N*sizeof(int));

 /*DSYEVR : Arguments   
=========   
    JOBZ    (input) CHARACTER*1   
            = 'N':  Compute eigenvalues only;   
            = 'V':  Compute eigenvalues and eigenvectors.   

    RANGE   (input) CHARACTER*1   
            = 'A': all eigenvalues will be found.   
            = 'V': all eigenvalues in the half-open interval (VL,VU]   
                   will be found.   
            = 'I': the IL-th through IU-th eigenvalues will be found.   
   ********* For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and   
   ********* DSTEIN are called   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the   
            leading N-by-N upper triangular part of A contains the   
            upper triangular part of the matrix A.  If UPLO = 'L',   
            the leading N-by-N lower triangular part of A contains   
            the lower triangular part of the matrix A.   
            On exit, the lower triangle (if UPLO='L') or the upper   
            triangle (if UPLO='U') of A, including the diagonal, is   
            destroyed.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    VL      (input) DOUBLE PRECISION   
    VU      (input) DOUBLE PRECISION   
            If RANGE='V', the lower and upper bounds of the interval to   
            be searched for eigenvalues. VL < VU.   
            Not referenced if RANGE = 'A' or 'I'.   

    IL      (input) INTEGER   
    IU      (input) INTEGER   
            If RANGE='I', the indices (in ascending order) of the   
            smallest and largest eigenvalues to be returned.   
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.   
            Not referenced if RANGE = 'A' or 'V'.   

    ABSTOL  (input) DOUBLE PRECISION   
            The absolute error tolerance for the eigenvalues.   
            An approximate eigenvalue is accepted as converged   
            when it is determined to lie in an interval [a,b]   
            of width less than or equal to   

                    ABSTOL + EPS *   max( |a|,|b| ) ,   

            where EPS is the machine precision.  If ABSTOL is less than   
            or equal to zero, then  EPS*|T|  will be used in its place,   
            where |T| is the 1-norm of the tridiagonal matrix obtained   
            by reducing A to tridiagonal form.   

            See "Computing Small Singular Values of Bidiagonal Matrices   
            with Guaranteed High Relative Accuracy," by Demmel and   
            Kahan, LAPACK Working Note #3.   

            If high relative accuracy is important, set ABSTOL to   
            DLAMCH( 'Safe minimum' ).  Doing so will guarantee that   
            eigenvalues are computed to high relative accuracy when   
            possible in future releases.  The current code does not   
            make any guarantees about high relative accuracy, but   
            future releases will. See J. Barlow and J. Demmel,   
            "Computing Accurate Eigensystems of Scaled Diagonally   
            Dominant Matrices", LAPACK Working Note #7, for a discussion   
            of which matrices define their eigenvalues to high relative   
            accuracy.   

    M       (output) INTEGER   
            The total number of eigenvalues found.  0 <= M <= N.   
            If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.   

    W       (output) DOUBLE PRECISION array, dimension (N)   
            The first M elements contain the selected eigenvalues in   
            ascending order.   

    Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M))   
            If JOBZ = 'V', then if INFO = 0, the first M columns of Z   
            contain the orthonormal eigenvectors of the matrix A   
            corresponding to the selected eigenvalues, with the i-th   
            column of Z holding the eigenvector associated with W(i).   
            If JOBZ = 'N', then Z is not referenced.   
            Note: the user must ensure that at least max(1,M) columns are   
            supplied in the array Z; if RANGE = 'V', the exact value of M   
            is not known in advance and an upper bound must be used.   
            Supplying N columns is always safe.   

    LDZ     (input) INTEGER   
            The leading dimension of the array Z.  LDZ >= 1, and if   
            JOBZ = 'V', LDZ >= max(1,N).   

    ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) )   
            The support of the eigenvectors in Z, i.e., the indices   
            indicating the nonzero elements in Z. The i-th eigenvector   
            is nonzero only in elements ISUPPZ( 2*i-1 ) through   
            ISUPPZ( 2*i ).   
   ********* Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1   

    WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= max(1,26*N).   
            For optimal efficiency, LWORK >= (NB+6)*N,   
            where NB is the max of the blocksize for DSYTRD and DORMTR   
            returned by ILAENV.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))   
            On exit, if INFO = 0, IWORK(1) returns the optimal LWORK.   

    LIWORK  (input) INTEGER   
            The dimension of the array IWORK.  LIWORK >= max(1,10*N).   

            If LIWORK = -1, then a workspace query is assumed; the   
            routine only calculates the optimal size of the IWORK array,   
            returns this value as the first entry of the IWORK array, and   
            no error message related to LIWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  Internal error   

*/  
 
  dsyevr('V', 'A', 'L', N, A, N, 0, 0, 0, 0, dlamch('S'), &M, W, Z, N, ISUPPZ, WORK, 26*N, IWORK, 10*N); // info in not here
    
/* allocate and initialise a new matrix B = Z*D */
  B = malloc(N*N*sizeof(double));
  
  printf("\nEigenvalues :");
  for (j = 0; j < N; ++j) {
       printf(" lam[%d] =  %4.2f  ",j, W[j]);
  }  
  printf("\n\n");

  for (j = 0; j < N; ++j) {
    for (i = 0; i < N; ++i) {
       printf(" exp(lam[%d][%d]) = exp(%4.2f)  ",i,j, W[j]);
    }       
    putchar('\n');
  }
  
  printf("\nEigenvector :\n");
  for (j = 0; j < N; ++j) {
    double  lambda = exp(W[j]);           // exponential
    for (i = 0; i < N; ++i) {
       B[i + N*j] = Z[i + N*j] * lambda;
       printf(" Z[%d][%d] = %7.5f",i, j, Z[i + N*j]);
    }       
    putchar('\n');
  }

/* calculate the product A = B*Z^T = Z*D*Z^T */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, Z, N, 0, A, N);
   
/* emit the result */
  printf("\nExponential matrix \n");
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {      
        printf(" A[%d][%d] = %2.1f",i, j, A[i + N*j]);
    }
    putchar('\n');
  }

  return 0;
}
/**************************************************************/
/*                 Call Creat Matrix                       *   /
/*************************************************************/
  double **CREAT_MATRIX(int rows, int cols) {
  int j; 
  double **mat; /* Defining a temporal "mat" array , otherwise would have to use (*memory) everywhere U is used (yuck) */

 /* Counting the times this function is called*/
 // static int IsFirstTime = 0; /* initialize the number of call */
 /* if(IsFirstTime == 0) {
       printf("CREAT TEMPORAL ARRAY : This function is called as a first time\n");
       IsFirstTime++; }
 else { 
       IsFirstTime++;
       printf("CREAT TEMPORAL ARRAY : This function has been called %d times so far\n", IsFirstTime);
 }
 */
 /* Each row should only contain double*, not double**, because each row will be an array of double */
  mat = (double **) malloc(cols * sizeof(double *)); // create Nx-row temporal pointers array 

  if (mat == NULL) {
    printf("Failure to allocate room for row pointers.\n ");
    exit(0);
  }

/* Had an error here.  Alloced rows above so iterate through rows not cols here */
  for (j = 0; j < rows; j++) {
/* Allocate array, store pointer  */
     mat[j] = (double *) malloc(rows * sizeof(double)); // Create on each temporal rows we allocate Ny-columns temporal pointer
    
      if (mat[j] == NULL) {
            printf("Failure to allocate for row[%d]\n", j);
            exit(0); 
/* still a problem here, if exiting with error, should free any column mallocs that were successful. */
       } 
  } 
  //printf("U array of zeros is pass by reference from Creat_matrix\n");
 /*for(i = 0; i < rows; i++) {
    for(j = 0; j < cols; j++) {
       printf("%f   ",mat[i][j]);
    }
    printf("\n");
  }      */

  return (mat); // return mat as U
  free(mat); // free mat No U
} /* END FUNCTION CREAT MATRIX */
