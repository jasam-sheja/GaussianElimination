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

#include <mpi.h>
#include "gaussian_elimination.hpp"



int my_rank;       /* process ID */
int PSIZE;         /* number of procs */

int gaussian_elimination_pref1();

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /* get proc ID */
    MPI_Comm_size(MPI_COMM_WORLD, &PSIZE);       /* get # of procs */


    gaussian_elimination_pref1();

    MPI_Finalize();
}

int gaussian_elimination_pref1(){
    

    int N = 6400; // TODO: read from args
    double *mat;    
    double* solution;
    
    if(my_rank==0) create_mat(N, mat);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    gaussian_elimination(mat, N, solution);
    

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();   
    ostringstream ss;
    ss << "time[N=" << N << ", PSize=" << PSIZE << "]: " << end-start << "s";
    if(my_rank==0) log(ss.str());
}
