/*///////////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2019 Mohamad Ammar Alsherfawi Aljazaerly (AKA Jasam Sheja)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//*/

#ifndef __UTILS_H__
#define __UTILS_H__

#include <mpi.h>
#include <string>
#include <stdlib.h>     /* srand, rand */

using namespace std;

void log(string msg, MPI_Comm comm=MPI_COMM_WORLD){
    int rank;
    MPI_Comm_rank(comm, &rank);     /* get proc ID */
    cout<<"["<<rank<<"]<< "<<msg<<endl;
}


#define AT(i,j) (i*M+j)

/*
*   create NxN+1 matrix for gaussian elimination
*/
template <class T>
void create_mat(const int N, T* &mat){
    const int M = N+1;
    mat = new T[N*M];
    for(int j=0;j<N;j++){
        mat[AT(j,N)] = ((T)rand()/RAND_MAX-0.5)*100;
        int max = -RAND_MAX;
        for(int i=0;i<N;i++){
            mat[AT(i,j)] = ((T)rand()/RAND_MAX-0.5)*100;
            if(max<mat[AT(i,j)]) max = mat[AT(i,j)];
        }
        // so the matrix won't need pivoting for gaussian elimination
        mat[AT(j,j)] = max+((T)rand()/RAND_MAX);
        mat[AT(j,j)] += (mat[AT(j,j)]==0)*((T)rand()/RAND_MAX); // Don't let the matrix be singular
    }
}

/*
*   create random linear system given the solution
*   note: this dosn't garantee independence
*/
template <class T>
void create_linear_system(T solution[], const int N, T* &mat){
    const int M = N+1;
    mat = new T[N*M];
    for(int j=0;j<N;j++){
        int max = -RAND_MAX;
        for(int i=0;i<N;i++){
            mat[AT(i,j)] = ((T)rand()/RAND_MAX-0.5)*100;
            if(max<mat[AT(i,j)]) max = mat[AT(i,j)];
        }
        // so the matrix won't need pivoting for gaussian elimination
        mat[AT(j,j)] = max+((T)rand()/RAND_MAX);
        mat[AT(j,j)] += (mat[AT(j,j)]==0)*((T)rand()/RAND_MAX); // Don't let the matrix be singular
    }
    for(int i=0;i<N;i++){
        mat[AT(i,N)] = 0;
        for(int j=0;j<N;j++){
            mat[AT(i,N)] += mat[AT(i,j)]*solution[j];
        }
    }
}

/*
*   print NxN+1 matrix 
*/
template <class T>
void print_mat(T* mat, const int N, const int _M=-1){
    int M = _M;
    if(M<0) M = N+1;
    std::ostringstream matout;
    matout << std::fixed << std::setprecision(2) ;
    matout << "\tMat[" << N << "x" << M << "]" << endl;
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++)
            matout << std::setw(10) << mat[AT(i,j)] << ", ";
        matout << endl;
    }
    log(matout.str());
}

/*
*   print gaussian eliminated NxN+1 matrix of system of equation
*/
template <class T>
void print_solution(T mat[], int N){
    print_mat(mat, 1, N);
}

#endif //__UTILS_H__