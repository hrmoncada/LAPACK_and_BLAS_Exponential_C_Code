/* expm_lapack_dgeev.c - Calculate the exponential of a general rectangular matrix A
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

/*  DGEEV Example.
   ==============
   Program computes the eigenvalues and left and right eigenvectors of a general rectangular matrix A:

A (Real array): 
    -1.01   0.86  -4.60   3.31  -4.81
     3.98   0.53  -7.04   5.29   3.55
     3.30   8.26  -3.89   8.20  -1.51
     4.43   4.96  -7.66  -7.33   6.18
     7.31  -6.43  -6.16   2.47   5.58

    DGEEV Example Program Results.
   ==============================

(WR, WI) Eigenvalues (Complex) :
 (  2.86, 10.76) (  2.86,-10.76) ( -0.69,  4.70) ( -0.69, -4.70) -10.46

(VL) Left eigenvectors (Complex array) :
 (  0.04,  0.29) (  0.04, -0.29) ( -0.13, -0.33) ( -0.13,  0.33)   0.04
 (  0.62,  0.00) (  0.62,  0.00) (  0.69,  0.00) (  0.69,  0.00)   0.56
 ( -0.04, -0.58) ( -0.04,  0.58) ( -0.39, -0.07) ( -0.39,  0.07)  -0.13
 (  0.28,  0.01) (  0.28, -0.01) ( -0.02, -0.19) ( -0.02,  0.19)  -0.80
 ( -0.04,  0.34) ( -0.04, -0.34) ( -0.40,  0.22) ( -0.40, -0.22)   0.18

(VR) Right eigenvectors (Complex array)
 (  0.11,  0.17) (  0.11, -0.17) (  0.73,  0.00) (  0.73,  0.00)   0.46
 (  0.41, -0.26) (  0.41,  0.26) ( -0.03, -0.02) ( -0.03,  0.02)   0.34
 (  0.10, -0.51) (  0.10,  0.51) (  0.19, -0.29) (  0.19,  0.29)   0.31
 (  0.40, -0.09) (  0.40,  0.09) ( -0.08, -0.08) ( -0.08,  0.08)  -0.74
 (  0.54,  0.00) (  0.54,  0.00) ( -0.29, -0.49) ( -0.29,  0.49)   0.16

(invVR) Right eigenvectors inverse  (Complex array)
 (0.21576,-0.32057) (0.74207, 0.34877)  (-0.36853, 0.67261)  (0.34700, 0.14645)  (0.13821, -0.43425)
 (0.21576, 0.32057) (0.74207, -0.34877) (-0.36853, -0.67261) (0.34700, -0.14645) (0.13821, 0.43425)
 (0.56090, 0.08732) (-0.25217,-1.07502) (0.26048, 0.58342)   (0.29978, -0.04027) (-0.19334, 0.71116)
 (0.56090,-0.08732) (-0.25217, 1.07502) (0.26048, -0.58342)  (0.29978, 0.04027)  (-0.19334, -0.71116)
 (0.05170, 0.00000) (0.70890, 0.00000)  (-0.16268, 0.00000)  (-1.00862, 0.00000) (0.23186, -0.00000)

expm(A) =  expm(Z*D*Z^-1) = Z*expm(D)*Z^-1 (Real array):
   -0.50217    4.56041    2.01829    2.38722   -0.98493
   -6.36228   -4.83109   12.28998   -2.47319   -6.76137
   -3.85152  -13.76345    6.27992   -6.46124   -2.03676
   -5.50963   -0.11275   10.85609   -0.33509   -6.46545
   -7.10558    3.60664   13.60548    1.03809   -8.66237

Compile and Link your code on Linux :

>> gcc expm_lapack_dgeev.c -lblas -llapack -lm  -L/usr/lib -o out

>> gcc expm_lapack_dgeev.c -lblas -llapack -lm  -o out

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
#include <complex.h>   
#include <tgmath.h>

/* Array size */
static const int N = 5; 

/* LU decomoposition of a general matrix */
extern void zgetrf_(int* M, int *N, double complex* A, int* lda, int* IPIV, int* INFO);

/* Generate inverse of a matrix given its LU decomposition */
extern void zgetri_(int* N, double complex* A, int* lda, int* IPIV, double complex* WORK, int* lwork, int* INFO);

/* Lapack library computes the eigenvalues and left and right eigenvectors of a general rectangular matrix*/
static int dgeev( char JOBVL, char JOBVR, int N, double *A, int LDA, double *WR, double *WI, double *VL, int LDVL, double *VR, int LDVR, double *WORK, int LWORK){

   extern void dgeev_( char *JOBVLp, char *JOBVRp, int *Np, double *A, int *LDAp, double *WR, double *WI, double *VL, int *LDVLp, double *VR, int *LDVRp,
                       double *WORK, int *LWORKp, int *INFOp );

   int INFO;

   dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);

   return INFO;
}

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double complex *a, int lda ) ;
extern void inverse_matrix(double complex* a, int n);
extern void print_real_matrix( char* desc, int m, int n, double* a, int lda );        /* input */
extern void print_complex_matrix( char* desc, int n, double* wi, double* v, int ldv );/* output */
extern void print_eigenvalues( char* desc, int n, double* wr, double* wi );
extern void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv );
double **CREAT_MATRIX();  

/* Main Program */
int main() {

   int i, j;
  
/* A must be symmetric semi-positive definit matrix*/  
  double **AA = CREAT_MATRIX (N , N);  //( rows, columns)
  AA[0][0] = -1.01;   AA[0][1] =  0.86;  AA[0][2] = -4.60;  AA[0][3] =  3.31;  AA[0][4] = -4.81;
  AA[1][0] =  3.98;   AA[1][1] =  0.53;  AA[1][2] = -7.04;  AA[1][3] =  5.29;  AA[1][4] =  3.55;
  AA[2][0] =  3.30;   AA[2][1] =  8.26;  AA[2][2] = -3.89;  AA[2][3] =  8.20;  AA[2][4] = -1.51;
  AA[3][0] =  4.43;   AA[3][1] =  4.96;  AA[3][2] = -7.66;  AA[3][3] = -7.33;  AA[3][4] =  6.18;
  AA[4][0] =  7.31;   AA[4][1] = -6.43;  AA[4][2] = -6.16;  AA[4][3] =  2.47;  AA[4][4] =  5.58;

/* Allocate memory space for the input/output parameters and workspace arrays */
   double *A = malloc(N*N*sizeof(double));
   double *WR = malloc(N*sizeof(double));
   double *WI = malloc(N*sizeof(double));
   double *VL = malloc(N*N*sizeof(double));
   double *VR = malloc(N*N*sizeof(double));
   double *WORK = malloc(N*N*sizeof(double));

/* Rearrage a 2D array A[M][N] into a 1D array A[M*N], Column major order*/  
   for (j = 0; j < N; j++) {
      for (i = 0;i < N; i++) A[i + N*j] = AA[i][j];
   }

/* Print A */
   print_real_matrix( "Matrix A", N, N, A, N );

/* Setup lapack library */
   dgeev('V', 'V', N, A, N, WR, WI, VL, N, VR, N, WORK, 4*N);   // info in not here 

   free(AA);
   free(A);
   free(WORK);

/* Print eigenvalues */
   print_eigenvalues( "Eigenvalues (lambda)", N, WR, WI );

/* Print left eigenvectors */
   print_eigenvectors( "Left eigenvectors (VI)", N, WI, VL, N );

/* Print right eigenvectors */
   print_eigenvectors( "Right eigenvectors (VR)", N, WI, VR, N );

/*  Allocate memory space for the eigenvalues matrix D = expm(lambda) = expm(WR + I*WI)*/
  double *expWR = malloc(N*sizeof(double));
  double *expWI = malloc(N*sizeof(double));

/* Allocate and initialise matrix B = VR*D */
  double complex *B = malloc(N*N*sizeof(double complex));

  for (j = 0; j < N; ++j) {
    expWR[j] = creal(exp(WR[j] + WI[j]*I));
    expWI[j] = cimag(exp(WR[j] + WI[j]*I));
  }
  free(WR);
  free(WI);
  free(VL);

/* print expm(lambda)*/
   print_eigenvalues( "expm(lambda)", N, expWR, expWI );

/* calculate the product B = VR*D , Column major order*/  
 for (i = 0; i < N; ++i) {
      j = 0;
      while( j < N ) {
        double exp_lambda_R = expWR[j]; //creal(exp(WR[j] + WI[j]*I));
        double exp_lambda_I = expWI[j]; //cimag(exp(WR[j] + WI[j]*I));
        if( WI[j] == (double)0.0 ) {
           B[i + N*j] = VR[i + N*j] * exp_lambda_R - VR[i + N*(j+1)] * exp_lambda_I ;  /* Real part*/
            j++;
         } else {
           B[i + N*j] = VR[i + N*j] * exp_lambda_R - VR[i + N*(j+1)] * exp_lambda_I  + I*(VR[i + N*j] * exp_lambda_I + VR[i + N*(j+1)] * exp_lambda_R); 
           B[i + N*(j+1)] = (VR[i + N*j] * exp_lambda_R - VR[i + N*(j+1)] * exp_lambda_I ) - I*(VR[i + N*j] * exp_lambda_I + VR[i + N*(j+1)] * exp_lambda_R);  
           j += 2;
         }
      }
}

/* Print B complex array, Column major order*/
   print_matrix( "Matrix B =  VR * expm(lambda)", N, N, B, N );

/* Allocate memory space for the Inverse matrix*/
   double complex *invVR =  malloc(N*N * sizeof(double complex));

/* *Rearrage VR array into a VR complex array, Column major order*/
   for( i = 0; i < N; i++ ) {
      j = 0;
      while( j < N ) {
         if( WI[j] == (double)0.0 ) {
            invVR[i+j*N] = VR[i+j*N];
             j++;
         } else {
            invVR[i+j*N] = VR[i+j*N] + I*VR[i+(j+1)*N] ;
            invVR[i+(j+1)*N] = VR[i+j*N] - I*VR[i+(j+1)*N] ;
            j += 2;
         }
      }
   }
   free(VR);

/* Find Inverse matrix of a complex array, Column major order*/        
    inverse_matrix(invVR, N);

/* Print Inverse matrix*/ 
    print_matrix( "Rigth Eigenvector Inverse array VR^(-1)", N, N, invVR, N );
 
/* Calculate the product A = B*inv(VR) = VR*D*inv(VR)  */
   double complex *C =  malloc(N*N * sizeof(double complex));
   double complex alpha, beta;
   int LDA, LDB, LDC, n, m ,k;

   alpha = 1.0, beta = 0.0;
   LDA = N, LDB = N, LDC = N, m = N, k = N, n = N;

   /* C =  alpha*A*B + beta*C = 1*B*invVR + 0*C*/
   cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, B, LDA, invVR, LDB, &beta, C, LDC);
   //cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, B, LDA, invVR, LDB, &beta, C, LDC);

/* Print the result */
   print_matrix("C = expm(A) = BB*VR^(-1)", N, N, C, N );
   free(invVR);
   free(B);

   //exit(0);    
  return 0;
}
/**************************************************************/
/*                        subprograms                        */
/*************************************************************/
/* Inverse complex matrix*/
void inverse_matrix(double complex *A, int N) {
    int *IPIV = malloc((N+1) * sizeof(int));
    int LWORK = N*N;
    double complex *WORK =  malloc(N*N * sizeof(double complex));
    int INFO;

/* Print details of LU factorization */
    zgetrf_(&N, &N, A, &N, IPIV, &INFO);
    //print_matrix( "Details of LU factorization", N, N, A, N );

/* Print details of inverse */
    zgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);
    //print_matrix( "Details of Invers", N, N, A, N );

    free(IPIV);/*free memory*/
    free(WORK);/*free memory*/
}

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double complex *a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " (%6.5f, %6.5f)", creal(a[i+j*lda]),cimag(a[i+j*lda]) );
                printf( "\n" );
        }
}

/* printing a matrix A */
void print_real_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
            for( j = 0; j < n; j++ ) {
               printf( " %6.5f", a[i+j*lda] );
            }
            printf( "\n" );
        }
}

/* printing a complex matrix A */
void print_complex_matrix( char* desc, int n, double* wi, double* v, int ldv ) {
   int i, j;
   printf( "\n %s\n", desc );
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            printf( " %6.5f", v[i+j*ldv] );
            j++;
         } else {
            printf( " (%6.5f,%6.5f)", v[i+j*ldv], v[i+(j+1)*ldv] );
            printf( " (%6.5f,%6.5f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
            j += 2;
         }
      }
      printf( "\n" );
   }
}

/* A 2D matrix A */
double ** CREAT_MATRIX(int rows , int cols) {
   int j;
   double **mat;
   
/* Allocations*/  
   mat = (double **) malloc(cols * sizeof(double *));

/* Check columns allocation*/
   if (mat == NULL){
      printf("Failire to allocate room for row pointers\n");
      exit(0);
   }

/* Check rows allocation*/
  for (j = 0; j < rows;j++){
     mat[j] = (double *) malloc(rows*sizeof(double));
     if (mat[j] == NULL){
         printf("Failire to allocate room for row[%d]\n",j);
         exit(0);
      }
   }
 return (mat); /*return mat array*/
 free (mat);   /*free memory*/
}

/* Auxiliary routine: printing eigenvalues */
void print_eigenvalues( char* desc, int n, double* wr, double* wi ) {
   int j;
   printf( "\n %s\n", desc );
   for( j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         printf( " %6.5f", wr[j] );
      } else {
         printf( " (%6.5f,%6.5f)", wr[j], wi[j] );
      }
   }
   printf( "\n" );
}

/* Auxiliary routine: printing eigenvectors */
void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv ) {
   int i, j;
   printf( "\n %s\n", desc );
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            printf( " %6.5f", v[i+j*ldv] );
            j++;
         } else {
            printf( " (%6.5f,%6.5f)", v[i+j*ldv], v[i+(j+1)*ldv] );
            printf( " (%6.5f,%6.5f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
            j += 2;
         }
      }
      printf( "\n" );
   }
}

