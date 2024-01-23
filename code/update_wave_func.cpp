//
// Created by phyzch on 1/23/24.
//
# include "util.h"
# include "molecule.h"
using namespace std;

void molecule:: update_real_part(vector<vector<double>> & wave_func_real){
    int i;
    int m;
    int vsize;
    // collect data for send_buffer.
    vsize = total_basis_set_num / num_proc;

    for(m=0; m < Haar_random_state_num; m++){
        for (i = 0; i < to_send_buffer_len; i++) {
            send_wave_func_real[m][i] = wave_func_real[m][tosendVecIndex[i] - my_id * vsize];
        }
        MPI_Alltoallv(&send_wave_func_real[m][0], tosendVecCount, tosendVecPtr, MPI_DOUBLE,
                      &recv_wave_func_real[m][0], remoteVecCount, remoteVecPtr, MPI_DOUBLE, MPI_COMM_WORLD);
        for(i=0;i<to_recv_buffer_len;i++){
            wave_func_real[m][i + basis_set_num]= recv_wave_func_real[m][i];
        }
    }
}

void molecule:: update_imag_part(vector<vector<double>> & wave_func_imag){
    int i;
    int vsize;
    int m;
    // collect data for send_buffer.
    vsize = total_basis_set_num / num_proc;

    for(m=0; m < Haar_random_state_num; m++){
        for (i = 0; i < to_send_buffer_len; i++) {
            send_wave_func_imag[m][i] = wave_func_imag[m][tosendVecIndex[i] - my_id * vsize];
        }
        MPI_Alltoallv(&send_wave_func_imag[m][0], tosendVecCount, tosendVecPtr, MPI_DOUBLE,
                      &recv_wave_func_imag[m][0], remoteVecCount, remoteVecPtr, MPI_DOUBLE, MPI_COMM_WORLD);
        for(i=0;i<to_recv_buffer_len;i++){
            wave_func_imag[m][i + basis_set_num]= recv_wave_func_imag[m][i];
        }
    }

}

void molecule:: update_individual_wave_func(vector<double> & wave_func){
    int i;
    int vsize;
    int m;
    // collect data for send_buffer.
    vsize = total_basis_set_num / num_proc;

    m = 0;
    for (i = 0; i < to_send_buffer_len; i++) {
        send_wave_func_imag[m][i] = wave_func[tosendVecIndex[i] - my_id * vsize];
    }
    MPI_Alltoallv(&send_wave_func_imag[m][0], tosendVecCount, tosendVecPtr, MPI_DOUBLE,
                  &recv_wave_func_imag[m][0], remoteVecCount, remoteVecPtr, MPI_DOUBLE, MPI_COMM_WORLD);
    for(i=0;i<to_recv_buffer_len;i++){
        wave_func [i+ basis_set_num]= recv_wave_func_imag[m][i];
    }
}