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

#ifndef __MPI_CYCLICSCATTER_H__
#define __MPI_CYCLICSCATTER_H__

#include <mpi.h>

#define MPI_CYCLICSCATTER_TAG 1025

/*
// Note: This implementation utilizes the recv buffer as temporary holder.
// Note: This implementation doesn't modify the send buffer.
*/

int MPI_CyclicScatter(
		    const void *sendbuf, 
		    int sendcount, 
		    int sendcyclesize,
		    MPI_Datatype sendtype,
		    void *recvbuf, 
	        int recvcount, 
	        int recvcyclesize,
		    MPI_Datatype recvtype, 
		    int root,
		    MPI_Comm comm){
		    
    int error_code;
	int rank, psize;
	MPI_Comm_rank(comm, &rank);     /* get proc ID */
    MPI_Comm_size(comm, &psize);    /* get # of procs */
	    
	if (((rank == root) && (sendcount == 0)) || ((rank != root) && (recvcount == 0)))
        return MPI_SUCCESS;
        
	//check args		
	if(sendcount%sendcyclesize) return MPI_ERR_COUNT; // cycle size doesn't fit buffer size
	if(recvcount%recvcyclesize) return MPI_ERR_COUNT; // cycle size doesn't fit buffer size
	if(sendcount%sendcyclesize*psize) return MPI_ERR_COUNT; // too many cycles
	if(recvcount%recvcyclesize*psize) return MPI_ERR_COUNT; // too many cycles
	
	int send_elem_size, recv_elem_size;
	MPI_Type_size(sendtype, &send_elem_size);
	MPI_Type_size(recvtype, &recv_elem_size);
	
    if(rank==root){
        int num_cycle = sendcount/sendcyclesize;
        for(int dst=0;dst<psize;dst++){
            if(dst==root)
                ;//local copy after loop
            else{
                for(int i=0;i<num_cycle;i++)
                    // char : cast void to byte size pointer
                    memcpy(((char*)recvbuf)+i*recvcyclesize*recv_elem_size, 
                           ((char*)sendbuf)+(dst+i*psize)*sendcyclesize*send_elem_size, 
                           recvcyclesize*recv_elem_size);
                           
                error_code = MPI_Send(recvbuf, sendcount, sendtype, dst, MPI_CYCLICSCATTER_TAG, comm);
                if(error_code!=MPI_SUCCESS) return error_code;
            }
        }             
        for(int i=0;i<num_cycle;i++)
            memcpy(((char*)recvbuf)+i*recvcyclesize*recv_elem_size, 
                   ((char*)sendbuf)+(root+i*psize)*sendcyclesize*send_elem_size, 
                   recvcyclesize*recv_elem_size);//local copy
    }else{
        MPI_Status status;
        error_code = MPI_Recv(recvbuf, recvcount, recvtype, root, MPI_CYCLICSCATTER_TAG, comm, &status);
        if(error_code!=MPI_SUCCESS) return error_code;
    }
    
    return MPI_SUCCESS;
}


#define MPI_CYCLICGATHER_TAG 1050

/*
// Note: This implementation utilizes the send buffer as temporary holder.
*/

int MPI_CyclicGather(
		    void *sendbuf, 
		    int sendcount, 
		    int sendcyclesize,
		    MPI_Datatype sendtype,
		    void *recvbuf, 
	        int recvcount, 
	        int recvcyclesize,
		    MPI_Datatype recvtype, 
		    int root,
		    MPI_Comm comm){
    int error_code;
    int rank, psize;
    MPI_Comm_rank(comm, &rank);      /* get proc ID */
    MPI_Comm_size(comm, &psize);     /* get # of procs */
        
    if (((rank == root) && (recvcount == 0)) || ((rank != root) && (sendcount == 0)))
        return MPI_SUCCESS;
            
    //check args		
	if(sendcount%sendcyclesize) return MPI_ERR_COUNT; // cycle size doesn't fit buffer size
	if(recvcount%recvcyclesize) return MPI_ERR_COUNT; // cycle size doesn't fit buffer size
	if(sendcount%sendcyclesize*psize) return MPI_ERR_COUNT; // too many cycles
	if(recvcount%recvcyclesize*psize) return MPI_ERR_COUNT; // too many cycles
	
	int send_elem_size, recv_elem_size;
	MPI_Type_size(sendtype, &send_elem_size);
	MPI_Type_size(recvtype, &recv_elem_size);
	
	if(rank==root){
	    int num_cycle = sendcount/sendcyclesize;
	    for(int i=0;i<num_cycle;i++)
            memcpy(((char*)recvbuf)+(root+i*psize)*sendcyclesize*send_elem_size, 
                   ((char*)sendbuf)+i*sendcyclesize*send_elem_size, 
                   sendcyclesize*send_elem_size);//local copy    
        for(int src=0;src<psize;src++){
            if(src==root)
                ;//local copy before loop
            else{
                MPI_Status status;           
                error_code = MPI_Recv(sendbuf, recvcount, recvtype, src, MPI_CYCLICGATHER_TAG, comm, &status);
                if(error_code!=MPI_SUCCESS) return error_code;
                for(int i=0;i<num_cycle;i++)
                    // char : cast void to byte size pointer
                    memcpy(((char*)recvbuf)+(src+i*psize)*recvcyclesize*recv_elem_size, 
                           ((char*)sendbuf)+i*sendcyclesize*send_elem_size, 
                           recvcyclesize*recv_elem_size); 
            }
        }             
        
    }else{
        error_code = MPI_Send(sendbuf, sendcount, sendtype, root, MPI_CYCLICGATHER_TAG, comm);
        if(error_code!=MPI_SUCCESS) return error_code;
    }
    
    return MPI_SUCCESS;
            
}

#endif
