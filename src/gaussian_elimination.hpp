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

#ifndef __MPI_GAUSSIAN_ELIMINATION_H__
#define __MPI_GAUSSIAN_ELIMINATION_H__

#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <iomanip>
#include "mpi_cyclic.hpp"

#ifdef LogDEBUG
#include "../utils.h"
#endif

using namespace std;

/*
*   Solve system of equations [A:b]
*/
#define GAUSSIAN_ELIMINATION_TAG 1025
void gaussian_elimination(double *mat, const int N, double* &solution, MPI_Comm comm=MPI_COMM_WORLD){

    int my_rank;       /* process ID */
    int PSIZE;         /* number of procs */
    
    MPI_Comm_rank(comm, &my_rank); /* get proc ID */
    MPI_Comm_size(comm, &PSIZE);       /* get # of procs */

    double* my_mat = new double[N*(N+1)/PSIZE];
    double *op_row, *_op_row = new double[N+1];
    
    MPI_CyclicScatter(mat, N*(N+1)/PSIZE, N+1, MPI_DOUBLE, my_mat, N*(N+1)/PSIZE, N+1, MPI_DOUBLE, 0, comm);

    #ifdef LogDEBUG
    log("My mat");
    print_mat(my_mat, N/PSIZE, N+1);
    #endif

    /* Forward elimination */
    for (int k=0; k<N-1; k++) {
        int row_idx = k/PSIZE*(N+1); // index at my_mat
        int owner = k%PSIZE;
        
        if(my_rank==owner){
            op_row = my_mat + row_idx;
        }else{
            op_row = _op_row;
        }
        MPI_Request req;
        MPI_Ibcast(op_row+k, N+1-k, MPI_DOUBLE, owner, comm, &req);
        if(my_rank!=owner) MPI_Wait(&req, MPI_STATUS_IGNORE);

        // forward elimination
        for(int i=k/PSIZE+(owner>=my_rank); i<N/PSIZE; i++){ // owner>=my_rank to order correspoding rows in my_mat
            double r = my_mat[i*(N+1)+k]/op_row[k]; //  A[i][k] / A[k][k];
            for(int j=k+1;j<N+1;j++) // start from k+1 because my_mat[i][k] would obviously be zero
                my_mat[i*(N+1)+j] -= r*op_row[j];
                
        }
    }

    #ifdef LogDEBUG
    log("Forward elimination");
    print_mat(my_mat, N/PSIZE, N+1);
    #endif
    
    /* Back substitution */
    for (int k=N-1; k>0; --k) {
        int row_idx = k/PSIZE*(N+1); // index at my_mat
        int owner = k%PSIZE;
        
        if(my_rank==owner){
            my_mat[row_idx+N] /= my_mat[row_idx+k]; // solution found x[k] = b[k]
            op_row = my_mat + row_idx;
        }else{
            op_row = _op_row;
        }
        MPI_Request req;
        MPI_Ibcast(op_row+N, 1, MPI_DOUBLE, owner, comm, &req);
        if(my_rank!=owner) MPI_Wait(&req, MPI_STATUS_IGNORE);
        
        // back substitution
        for(int i=0; i<=k/PSIZE-(owner<=my_rank); i++){
            my_mat[i*(N+1)+N] -= my_mat[i*(N+1)+k] * op_row[N];
        }
    }
    // k == 0
    if(my_rank==0){
        my_mat[N] /= my_mat[0]; // solution found x[k] = b[k]
    }

    #ifdef LogDEBUG
    log("Back substitution");
    print_mat(my_mat, N/PSIZE, N+1);
    #endif
    
    solution = new double[N];
    double* my_solution = new double[N/PSIZE];
    for(int i=0;i<N/PSIZE;i++)
        my_solution[i] = my_mat[i*(N+1)+N];
    
    //gather b's values which are the solution;
    MPI_CyclicGather(my_solution, N/PSIZE, 1, MPI_DOUBLE, solution, N/PSIZE, 1, MPI_DOUBLE, 0, comm);
}
#endif
