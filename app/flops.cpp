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
#include <sstream>
#include "gaussian_elimination.hpp"
#include "utils.h"



int my_rank;       /* process ID */
int PSIZE;         /* number of procs */

double timeit(double *mat, int n);
double flop(double n);

const int MAX_N = 6400;
const int segments = 64;

int main(int argc, char* argv[]) {


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /* get proc ID */
    MPI_Comm_size(MPI_COMM_WORLD, &PSIZE);       /* get # of procs */

    int N = (MAX_N/PSIZE)*PSIZE;
    double *mat;
    for(int s=1;s<=segments;s++){
        int n = s*N/segments;
        create_mat(n, mat);

        double num_flop = flop(n);
        double time = timeit(mat, n); 
        
        if(my_rank == 0){
            ostringstream ss;
            ss << "{";
            ss << "\"n\":" << n << ", ";
            ss << "\"PSIZE\":" << PSIZE << ", ";
            ss << "\"flop\":" << num_flop << ", ";
            ss << "\"time\":" << time << ", ";
            ss << "\"flops\":" << num_flop/time;
            ss << "}";
            log(ss.str());
        } 

        delete mat;
    }

    MPI_Finalize();
}

double flop(double n){
    double res = 0;
    res += n*(n-1)*(4*n+1)/6; //#forward elimination
    res += n*n;               //#backward elimination
    return res;
}

double timeit(double *mat, int n){
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    //measured functionality
    {
        double *my_solution;
        gaussian_elimination(mat, n, my_solution, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();  

    return end-start;
}
