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

double parallel(double *mat, int n);
double sequential(double *mat, int n);

const int MAX_N = 6000;
const int repeat = 1;

int main(int argc, char* argv[]) {


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /* get proc ID */
    MPI_Comm_size(MPI_COMM_WORLD, &PSIZE);       /* get # of procs */

    int n = (MAX_N/PSIZE)*PSIZE;
    double *mat;
    for(int i=0;i<repeat;i++){
        create_mat(n, mat);

        double parallel_timeit = parallel(mat, n); 
        double sequential_timeit = sequential(mat, n); 
        
        if(my_rank == 0){
            ostringstream ss;
            ss << "{";
            ss << "\"n\":" << n << ", ";
            ss << "\"PSIZE\":" << PSIZE << ", ";
            ss << "\"parallel\":" << parallel_timeit << ", ";
            ss << "\"sequential\":" << sequential_timeit;
            ss << "}";
            log(ss.str());
        } 

        delete mat;
    }

    MPI_Finalize();
}


double parallel(double *mat, int n){
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

double sequential(double *mat, int n){
    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    const int ranks[1] = {0};

    // Construct a group containing all of the prime ranks in world_group
    MPI_Group root_group;
    MPI_Group_incl(world_group, 1, ranks, &root_group);

    // Create a new communicator based on the group
    MPI_Comm root_comm;
    MPI_Comm_create(MPI_COMM_WORLD, root_group, &root_comm);

    double start = MPI_Wtime();

    //measured functionality
    if(my_rank==0){
        double *my_solution;
        gaussian_elimination(mat, n, my_solution, root_comm);
    }
    double end = MPI_Wtime();  

    MPI_Group_free(&world_group);
    MPI_Group_free(&root_group);
    if(root_comm!=MPI_COMM_NULL)
    MPI_Comm_free(&root_comm);

    return end-start;
}
