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

int scatter_pref();

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /* get proc ID */
    MPI_Comm_size(MPI_COMM_WORLD, &PSIZE);       /* get # of procs */


    scatter_pref();

    MPI_Finalize();
}



int _blk_scatter_pref(int* array, int* Harray, const int N){
     MPI_Barrier(MPI_COMM_WORLD);
     double start = MPI_Wtime();
     
     //print_mat(array, N, N+1); 
     MPI_CyclicScatter(array, N*(N+1)/2, (N+1), MPI_INT, Harray, N*(N+1)/2, (N+1), MPI_INT, 0, MPI_COMM_WORLD);
     //print_mat(Harray, N/2, N+1);  
     
     MPI_Barrier(MPI_COMM_WORLD);
     double end = MPI_Wtime();   
     ostringstream ss;
     ss << "delay[blk]: " << end-start << "s";
     if(my_rank==0) log(ss.str());
}

int _ins_scatter_pref(int* array, int* Harray, const int N){
     MPI_Barrier(MPI_COMM_WORLD);
     double start = MPI_Wtime();
     
     //print_mat(array, N, N+1); 
     MPI_ICyclicScatter(array, N*(N+1)/2, (N+1), MPI_INT, Harray, N*(N+1)/2, (N+1), MPI_INT, 0, MPI_COMM_WORLD);
     //print_mat(Harray, N/2, N+1);  
     
     MPI_Barrier(MPI_COMM_WORLD);
     double end = MPI_Wtime();   
     ostringstream ss;
     ss << "delay[ins]: " << end-start << "s";
     if(my_rank==0) log(ss.str());
}
int scatter_pref(void)
{
    const int N = 10000;
    int* array = new int[N*(N+1)];
    int* Harray = new int[N*(N+1)/2];
    if(my_rank==0)
        for(int i=0;i<N;++i)
            for(int j=0;j<=N;++j)
                array[i*(N+1)+j] = i*(N+1)+j;
                
     delete[] array;
     create_mat(N, array);
     _ins_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _blk_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _ins_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _blk_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _ins_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _blk_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _ins_scatter_pref(array, Harray, N);
     
     delete[] array;
     create_mat(N, array);
     _blk_scatter_pref(array, Harray, N);
     return 0;
}