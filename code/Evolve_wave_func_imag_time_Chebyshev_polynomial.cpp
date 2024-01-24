//
// Created by phyzch on 1/1/23.
//
# include "util.h"
# include "molecule.h"
using namespace std;

void molecule::update_polyn23( double ** & send_polyn,
                               double ** & recv_polyn,
                               vector<vector<double>> &  Chebyshev_polyn) {
    // update Chebychev_polyn[2] and Chebychev_polyn[3]
    int i;
    int vsize;
    int begin_index;
    // collect data for send_buffer.
    vsize = total_basis_set_num /num_proc;
    begin_index = vsize * my_id;
    for(i = 0;i < to_send_buffer_len; i++){
        send_polyn[2][i] = Chebyshev_polyn[2][tosendVecIndex[i] - begin_index];
        send_polyn[3][i] = Chebyshev_polyn[3][tosendVecIndex[i] - begin_index];
    }

    MPI_Alltoallv(&send_polyn[2][0],tosendVecCount,tosendVecPtr,MPI_DOUBLE,
                  &recv_polyn[2][0],remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Alltoallv(&send_polyn[3][0],tosendVecCount,tosendVecPtr,MPI_DOUBLE,
                  &recv_polyn[3][0],remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);

    for(i=0;i<to_recv_buffer_len;i++){
        Chebyshev_polyn[2][ i + basis_set_num ]  = recv_polyn[2][i];
        Chebyshev_polyn[3][ i + basis_set_num ]  = recv_polyn[3][i];
    }
}


void molecule::prepare_Chebyshev_polynomial_evolution_imag_time( ){
    // except to call this function, we also have to call prepare_evolution function to prepare parallel computation : remote_Vec_count etc.
    // this function should be called after prepare_evolution() function
    // we evolve each step with delt.
    int i, j;
    double max_potential, min_potential;
    max_potential = basis_set_potential_energy_all[0] ;
    min_potential = basis_set_potential_energy_all[0];

    for(i=0;i<total_basis_set_num;i++){
        if (max_potential < basis_set_potential_energy_all[i] ){
            max_potential = basis_set_potential_energy_all[i];
        }
        if (min_potential > basis_set_potential_energy_all[i]){
            min_potential = basis_set_potential_energy_all[i];
        }
    }

    imag_Chebyshev_R  = (max_potential - min_potential ) * 0.55 ; // make R slighly larger to ensure omega < 1.
    imag_Chebyshev_e0 = (max_potential + min_potential) / 2; // imag_Chebyshev_R + min_potential


    imag_time_shifted_mat = new double [mat_num];

    for(i=0;i< mat_num ;i++){
        if( irow[i] == icol[i]){
            imag_time_shifted_mat[i] = mat[i] - imag_Chebyshev_e0;
        }
        else{
            imag_time_shifted_mat[i] = mat[i];
        }
        imag_time_shifted_mat[i] = imag_time_shifted_mat[i] / imag_Chebyshev_R;
    }

    imag_Chebyshev_prefactor = std::exp(- delt * imag_Chebyshev_e0);
    imag_Chebyshev_R_times_imag_time = imag_Chebyshev_R * delt;  // used to decide order of Chebyshev polynomial we use.

    imag_N_chebyshev = ceil(1.5 * imag_Chebyshev_R_times_imag_time);
    int N_chebyshev_minimum = 10;
    if(imag_N_chebyshev < N_chebyshev_minimum ){
        imag_N_chebyshev = N_chebyshev_minimum;
    }
    if(my_id == 0){
        cout << "Using Chebyshev method to compute Boltzmann factor. delt * R " << imag_Chebyshev_R_times_imag_time << endl;
        log << "Using Chebyshev method to compute Boltzmann factor. delt * R = " << imag_Chebyshev_R_times_imag_time << endl;
        cout << "Using Chebychev method.  order of Chebychev polynomial cutoff = " << imag_N_chebyshev << endl;
        log << "Using Chebychev method.  order of Chebychev polynomial cutoff = " << imag_N_chebyshev << endl;
    }

    imag_Bessel_function_array = new double [imag_N_chebyshev + 1];
    for(i=0; i <= imag_N_chebyshev; i++){
        imag_Bessel_function_array[i] = std::cyl_bessel_i(i, imag_Chebyshev_R_times_imag_time) ;  // modified Bessel function of first kind I_{i}(multiple_wave_func_real)
    }

    // Used for recursive solving T_{n}(multiple_wave_func_real) : T_{n+1}(multiple_wave_func_real) = 2* multiple_wave_func_real * T_{n}(multiple_wave_func_real) - T_{n-1}(multiple_wave_func_real)
    for(i=0;i<6;i++){
        vector <double> v ( basis_set_num + to_recv_buffer_len , 0 );
        imag_time_Chebyshev_polyn.push_back(v);
    }

    imag_send_polyn = new double * [6];
    imag_recv_polyn = new double * [6];
    for(i=0; i<6 ;i++ ){
        imag_send_polyn[i] = new double [to_send_buffer_len];
        imag_recv_polyn[i] = new double [to_recv_buffer_len];
    }

}

void molecule:: free_space_for_imag_Chebyshev_method(){
    delete [] imag_time_shifted_mat;
    delete [] imag_Bessel_function_array;

    int i;
    for(i=0;i<6;i++){
        delete [] imag_send_polyn[i];
        delete [] imag_recv_polyn[i];
    }
    delete [] imag_send_polyn;
    delete [] imag_recv_polyn;
}

void molecule::Chebyshev_method_imag_time_multiple_wave_func(const vector<vector<double>> & wave_func_real , const vector<vector<double>> & wave_func_imag,
                                                             vector<vector<double>> & imag_time_evolved_wave_func_real, vector<vector<double>> & imag_time_evolved_wave_func_imag , double imag_time){
    int i,j;
    int step_number;
    step_number = int(imag_time / delt);


    imag_time_evolved_wave_func_real.clear();
    imag_time_evolved_wave_func_imag.clear();

    for(i=0; i< Haar_random_state_num; i++){
        vector<double> Boltzmann_factor_weighted_wave_func_real_element;
        vector<double> Boltzmann_factor_weighted_wave_func_imag_element;

        vector<double> wave_func_real_copy = wave_func_real[i];
        vector<double> wave_func_imag_copy = wave_func_imag[i];

        // for each time step, we evolve e^{- H dt}
        for(j=0; j<step_number; j++){
            Chebyshev_method_imag_time_single_wavefunc(wave_func_real_copy, wave_func_imag_copy);
        }

        Boltzmann_factor_weighted_wave_func_real_element = wave_func_real_copy;
        Boltzmann_factor_weighted_wave_func_imag_element = wave_func_imag_copy;


        imag_time_evolved_wave_func_real.push_back(Boltzmann_factor_weighted_wave_func_real_element);
        imag_time_evolved_wave_func_imag.push_back(Boltzmann_factor_weighted_wave_func_imag_element);
    }


//    if(my_id ==0){
//        cout << "finish Chebyshev imag time evolution" << endl;
//    }
}

void molecule::Chebyshev_method_imag_time_single_wavefunc(vector<double> & wave_func_real , vector<double> & wave_func_imag ){
    // compute Bolzmann weighted wave function .
    // input :: one_fourth_beta,  multiple_wave_func_real,  wave_func_imag
    // output :: Boltzmann_factor_weighted_wave_func_x ,  Boltzmann_factor_weighted_wave_func_y
    // caution: before call this function, make sure update wave function component from other process.
    // have to update

    int i, j, k , m;
    int irow_index , icol_index;
    double bess;
    double prefactor;

    update_individual_wave_func(wave_func_real);
    update_individual_wave_func(wave_func_imag);

    vector<double> & creal =  wave_func_real;
    vector<double> & cimag = wave_func_imag ;


    // imag_time_Chebyshev_polyn is Chebyshev polynomial of Hamiltonian act upon wave function
    // 0,2,4 for real part. 1,3,5 for imag part
    for(i=0;i< basis_set_num + to_recv_buffer_len; i++) {
        imag_time_Chebyshev_polyn[0][i] = creal[i];
        imag_time_Chebyshev_polyn[1][i] = cimag[i];
        imag_time_Chebyshev_polyn[2][i] = 0;
        imag_time_Chebyshev_polyn[3][i] = 0;
        imag_time_Chebyshev_polyn[4][i] = 0;
        imag_time_Chebyshev_polyn[5][i] = 0;
    }
    // zeroth order
    bess = imag_Bessel_function_array[0];
    prefactor = imag_Chebyshev_prefactor * bess;  // Chebyshev prefactor : e^{-\beta E_{0}} . bess = I_{0}(R)
    for(i=0;i< basis_set_num ;i++){
        creal[i] = prefactor * imag_time_Chebyshev_polyn[0][i];
        cimag[i] = prefactor * imag_time_Chebyshev_polyn[1][i];
    }

    // first order
    bess = imag_Bessel_function_array[1];
    prefactor = 2 * bess * imag_Chebyshev_prefactor;
    for(i=0;i< mat_num;i++){  // \omega =  - shifted_dmat
        irow_index = local_irow[i];
        icol_index = local_icol[i];
        imag_time_Chebyshev_polyn[2][irow_index ] = imag_time_Chebyshev_polyn[2][irow_index] + (-imag_time_shifted_mat[i]) * imag_time_Chebyshev_polyn[0][icol_index];
        imag_time_Chebyshev_polyn[3][irow_index ] = imag_time_Chebyshev_polyn[3][irow_index] + (-imag_time_shifted_mat[i]) * imag_time_Chebyshev_polyn[1][icol_index];
    }

    // used for communication between different process
    update_polyn23(imag_send_polyn, imag_recv_polyn, imag_time_Chebyshev_polyn);
    for(i=0;i<basis_set_num;i++){
        creal[i] = creal[i] + prefactor * imag_time_Chebyshev_polyn[2][i];
        cimag[i] = cimag[i] + prefactor * imag_time_Chebyshev_polyn[3][i];
    }

    // Remaining terms :
    for(k=2; k <= imag_N_chebyshev; k++){
        bess = imag_Bessel_function_array[k];
        prefactor = imag_Chebyshev_prefactor * bess * 2 ;

        // Use Chebyshev polynomial relationship  T_{k+2}(multiple_wave_func_real) = 2 * multiple_wave_func_real *  T_{k+1}(multiple_wave_func_real) - T_{k}(multiple_wave_func_real)
        for(i=0;i< mat_num ; i++ ){
            irow_index = local_irow[i];
            icol_index = local_icol[i];
            imag_time_Chebyshev_polyn[4][irow_index] = imag_time_Chebyshev_polyn[4][irow_index] + 2 * (-imag_time_shifted_mat[i]) * imag_time_Chebyshev_polyn[2][icol_index];
            imag_time_Chebyshev_polyn[5][irow_index] = imag_time_Chebyshev_polyn[5][irow_index] + 2 * (-imag_time_shifted_mat[i]) * imag_time_Chebyshev_polyn[3][icol_index];
        }
        for(i=0;i<basis_set_num;i++){
            imag_time_Chebyshev_polyn[4][i] = imag_time_Chebyshev_polyn[4][i] - imag_time_Chebyshev_polyn[0][i];
            imag_time_Chebyshev_polyn[5][i] = imag_time_Chebyshev_polyn[5][i] - imag_time_Chebyshev_polyn[1][i];
        }

        // update dx , dy
        for(i=0;i<basis_set_num;i++){
            creal[i] = creal[i]  + prefactor * imag_time_Chebyshev_polyn[4][i];
            cimag[i] = cimag[i] + prefactor * imag_time_Chebyshev_polyn[5][i];
        }

        for(i=0;i<basis_set_num;i++){
            imag_time_Chebyshev_polyn[0][i] = imag_time_Chebyshev_polyn[2][i];
            imag_time_Chebyshev_polyn[1][i] = imag_time_Chebyshev_polyn[3][i];
            imag_time_Chebyshev_polyn[2][i] = imag_time_Chebyshev_polyn[4][i];
            imag_time_Chebyshev_polyn[3][i] = imag_time_Chebyshev_polyn[5][i];
            imag_time_Chebyshev_polyn[4][i] = 0;
            imag_time_Chebyshev_polyn[5][i] = 0;
        }
        update_polyn23(imag_send_polyn, imag_recv_polyn, imag_time_Chebyshev_polyn);

    }



}
