//
// Created by phyzch on 12/29/22.
//
# include "util.h"
# include "molecule.h"
using namespace std;


int compare(const void * a, const void * b){
    return *(int *) a - * (int *)b;
}

int  molecule::construct_receive_buffer_index(int * remoteVecCount_element, int * remoteVecPtr_element, int * remoteVecIndex_element){
    // input: remoteVecCount: total number of element need to receive from each process.
    //        remoteVecPtr: displacement in remoteVecIndex for element in each process.
    //        remoteVecIndex: index for remote vector we need to receive. (they may allocate in different remote process.)
    // return: length of remoteVecIndex.

    int i,j;
    int total_remoteVecCount;
    // range for element in process is [local_begin, local_end)
    int vsize = total_basis_set_num/ num_proc;
    int local_begin = total_basis_set_num/ num_proc * my_id;
    int local_end;
    int remote_pc_id;
    if(my_id != num_proc-1) {
        local_end = total_basis_set_num / num_proc * (my_id + 1);
    }
    else{
        local_end = total_basis_set_num;
    }
    // ---------------------------------------------------------------
    vector <int> col_index_copy = icol;  // column index in Hamiltonian has nonzero element.
    sort(col_index_copy.begin(),col_index_copy.end()); // sort vector.
    int col_array_size = col_index_copy.size();
    int prev_col=-1;
    j=0;

    // col_index_copy is sorted. But there is chance we have the same element.
    for(i=0;i<col_array_size;i++){
        if( (col_index_copy[i]>prev_col)   and ( (col_index_copy[i]<local_begin) or (col_index_copy[i] >= local_end) )  ){
            // this matrix element is not in process.
            if (col_index_copy[i] >= vsize * (num_proc-1) ){
                remote_pc_id = num_proc-1;
            }
            else{
                remote_pc_id = col_index_copy[i] / vsize;
            }
            remoteVecCount_element[remote_pc_id] ++;
            remoteVecIndex_element [j] = col_index_copy[i];  // vector index need to receive. (global index , ordered)
            j++;
        }
        prev_col= col_index_copy[i];
    }
    remoteVecPtr_element[0]=0;   // displacement for remote vector from each process in remoteVecIndex.
    for(i=1;i<num_proc;i++){
        remoteVecPtr_element[i] = remoteVecPtr_element[i-1] + remoteVecCount_element[i-1];
    }

    total_remoteVecCount = 0;
    for(i=0;i<num_proc;i++){
        total_remoteVecCount = total_remoteVecCount + remoteVecCount_element[i];
    }
    return total_remoteVecCount;
}

int construct_send_buffer_index(int * remoteVecCount_element, int * remoteVecPtr_element, int * remoteVecIndex_element,
                                int * tosendVecCount_element, int * tosendVecPtr_element, int* & tosendVecIndex_ptr){
    //  tosend_Vec_count record number of element to send to each process.
    // tosend_Vec_Index record the global index of vector the process have to send
    // tosend_Vec_Ptr record the offset of vector to send to each other process.
    // return to_send_buffer_len: lenfth of tosendVecIndex
    int i;
    int to_send_buffer_len;

    MPI_Alltoall(&remoteVecCount_element[0],1,MPI_INT,&(tosendVecCount_element[0]),1,MPI_INT,MPI_COMM_WORLD);

    // compute displacement for each process's data.
    tosendVecPtr_element[0]=0;
    for(i=1;i<num_proc;i++){
        tosendVecPtr_element[i]= tosendVecPtr_element[i-1] + tosendVecCount_element[i-1];
    }
    // compute total length of buffer to send
    to_send_buffer_len=0;
    for(i=0;i<num_proc;i++){
        to_send_buffer_len = to_send_buffer_len + tosendVecCount_element[i];
    }
    // Index (in global) of element to send. use MPI_Alltoallv to receive the index to send.
    tosendVecIndex_ptr = new int [to_send_buffer_len];
    MPI_Alltoallv(&remoteVecIndex_element[0],remoteVecCount_element,remoteVecPtr_element,MPI_INT,
                  & tosendVecIndex_ptr[0],tosendVecCount_element,tosendVecPtr_element,MPI_INT,MPI_COMM_WORLD);

    return to_send_buffer_len;
}


void molecule::prepare_evolution(){
    // compute buffer to receive and send for each process.
    // resize xd,yd to provide extra space for recv_buffer.
    // allocate space for send_wave_func_real , send_wave_func_imag buffer.
    // Index for remoteVecIndex, tosendVecIndex are computed here.
    int m,i;
    int vsize;
    // Index for vector to send and receive.
    // remoteVecCount: total number to receive. remoteVecPtr: displacement in remoteVecIndex for each process. remoteVecIndex: index in other process to receive.
    // tosendVecCount: total number to send to other process. tosendVecPtr: displacement in tosendVecIndex in each process.  tosendVecIndex: Index of element in itself to send. (it's global ,need to be converted to local index)

    //------------------Allocate space for vector to receive ---------------------
    remoteVecCount = new int [num_proc];
    remoteVecPtr = new int [num_proc];
    remoteVecIndex = new int [mat_num];
    for(i=0;i<num_proc;i++){
        remoteVecCount[i] = 0;
    }


    tosendVecCount = new int [num_proc];
    tosendVecPtr = new int [num_proc];


    int * search_Ind; // local variable, used for compute local_icol;
    int col_index_to_search;
    // local column index used when we do H *random_wave_func_real and H*random_wave_func_imag

    // buffer to send and receive buffer to/from other process.
    recv_wave_func_real= new double * [Haar_random_state_num];
    recv_wave_func_imag= new double * [Haar_random_state_num];
    send_wave_func_real= new double * [Haar_random_state_num];
    send_wave_func_imag = new double *[Haar_random_state_num];

    vsize = total_basis_set_num / num_proc;
    to_recv_buffer_len = construct_receive_buffer_index(remoteVecCount,remoteVecPtr,
                                                        remoteVecIndex);  // construct buffer to receive.
    to_send_buffer_len = construct_send_buffer_index(remoteVecCount, remoteVecPtr, remoteVecIndex,
                                                     tosendVecCount, tosendVecPtr, tosendVecIndex);

    for(m=0; m < Haar_random_state_num; m++){
        random_wave_func_real[m].resize(basis_set_num + to_recv_buffer_len);
        random_wave_func_imag[m].resize(basis_set_num + to_recv_buffer_len);
        recv_wave_func_real[m] = new double [to_recv_buffer_len];
        recv_wave_func_imag[m]= new double [to_recv_buffer_len];
        send_wave_func_real[m]= new double [to_send_buffer_len];
        send_wave_func_imag[m] = new double [to_send_buffer_len];
    }


    // construct local_irow, local_icol
    local_irow.reserve(mat_num);
    local_icol.reserve(mat_num);
    for(i=0;i<mat_num;i++){
        local_irow.push_back(irow[i] - my_id * vsize);  // set local index for row index
        col_index_to_search= icol[i];
        search_Ind=(int *) bsearch(&col_index_to_search, remoteVecIndex, to_recv_buffer_len, sizeof(int), compare);
        if(search_Ind!=NULL){
            // this column index is not in local matrix, and we should get it from other process (remoteVec)
            local_icol.push_back(basis_set_num + (search_Ind - remoteVecIndex) );
        }
        else{ // this column index is in local matrix.
            local_icol.push_back (icol[i] - my_id * vsize );
        }
    }


}