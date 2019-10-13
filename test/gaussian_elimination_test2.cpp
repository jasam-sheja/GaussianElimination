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

int gaussian_elimination_test2();

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /* get proc ID */
    MPI_Comm_size(MPI_COMM_WORLD, &PSIZE);       /* get # of procs */

    gaussian_elimination_test2();

    MPI_Finalize();
}

int _gaussian_elimination_test(const int n){
    double* mat;
    double solution[n];
    for(int i=0;i<n;i++)
        solution[i] = i*2-i*i/n; // some solution
    create_linear_system(solution, n, mat);

    double *my_solution;
    gaussian_elimination(mat, n, my_solution);
    if(my_rank != 0) return 1;
    #ifdef LogDEBUG
    if(n<10) {
        print_solution(solution, n);
        print_mat(mat, n);
        print_solution(my_solution, n);
    }
    #endif
    for(int i=0;i<n;i++)
        if(abs(my_solution[i]-solution[i])>1e-2)
            return 0;
    return 1;
}

int gaussian_elimination_test2(){
    int fails(0), total(0);
    for(int n=PSIZE;n<=(1<<10);n<<=1, total++)
        fails += 1-_gaussian_elimination_test(n);
    if(my_rank == 0){
        ostringstream ss;
        ss << "number of failures: " << fails << " out of " << total;
        log(ss.str());
    }
}