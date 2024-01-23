//
// Created by phyzch on 1/2/23.
//
# include "util.h"
# include "molecule.h"
using namespace std;

void molecule::prepare_Chebyshev_polynomial_evolution_real_time(){
    // except to call this function, we also have to call prepare_evolution function to prepare parallel computation : remote_Vec_count etc.
    // this function should be called after prepare_evolution() function
    // we evolve each step with delt.
    int i,j;
    double max_potential, min_potential;

    // compute maximum potential energy and minimum potential energy.
    max_potential = basis_set_potential_energy_all[0];
    min_potential = basis_set_potential_energy_all[0];

    for(i=0;i<total_basis_set_num;i++){
        if (max_potential < basis_set_potential_energy_all[i] ){
            max_potential = basis_set_potential_energy_all[i];
        }
        if (min_potential > basis_set_potential_energy_all[i]){
            min_potential = basis_set_potential_energy_all[i];
        }
    }

    // range of potential: [Vmax - Vmin] * 0.55. (It supposes to be [Vmax-Vmin]/2, but we make it 0.55 to make sure it converges]
    real_Chebyshev_R = (max_potential - min_potential ) * 0.55;
    // [Vmax + Vmin]/2.
    real_Chebyshev_e0 = (max_potential + min_potential ) / 2;

    // [H - E]/R: normalized Hamiltonian.
    real_time_normalized_mat = new double [mat_num];
    for(i=0;i<mat_num;i++){
        if(irow[i] == icol[i]){
            real_time_normalized_mat[i] = mat[i] - real_Chebyshev_e0;
        }
        else{
            real_time_normalized_mat[i] = mat[i];
        }
        real_time_normalized_mat[i] = real_time_normalized_mat[i] / real_Chebyshev_R;
    }

    real_Chebyshev_R_t = real_Chebyshev_R * delt;
    real_Chebyshev_expr = std::cos(delt * real_Chebyshev_e0); // real part of exp(i[Vmax + Vmin]/2 * t)
    real_Chebyshev_expi = std::sin(delt * real_Chebyshev_e0); // imaginary part of exp( i[Vmax + Vmin] /2 *t)

    // order of Chebyshev polynomial to compute to ensure convergence.
    int N_Chebyshev_minimum = 10;
    real_N_chebyshev = ceil(1.5 * real_Chebyshev_R_t);
    if(real_N_chebyshev < N_Chebyshev_minimum){
        real_N_chebyshev = N_Chebyshev_minimum;
    }

    if(my_id == 0){
        cout << "Using Chebychev method in real time  R * dt=   " << real_Chebyshev_R_t << endl;
        log << "Using Chebychev method in real time  R * dt=   " << real_Chebyshev_R_t << endl;
        cout << "Using Chebychev method in real time. order of Chebychev polynomial:    " << real_N_chebyshev<< endl;
        log << "Using Chebychev methodin real time. order of Chebychev polynomial:     " << real_N_chebyshev << endl;
    }

    real_Bessel_function_array = new double [real_N_chebyshev + 1];
    for(i = 0;i <= real_N_chebyshev; i++){
        real_Bessel_function_array[i] = std::cyl_bessel_j(i, real_Chebyshev_R_t);
        // cyl_bessel_j: bessel function of the first kind of order given by i.
        // See the page 3968 in J. Chem. Phys. 81, 3967 (1984)
    }

    // for using recursive relation of Chebyshev polynomial to compute wave function.
    for(i = 0;i < 6; i++){
        vector<double> v (basis_set_num + to_recv_buffer_len,0);
        real_time_Chebyshev_polyn.push_back(v);
    }

    // for use of MPI to communicate the wave function array.
    real_send_polyn = new double * [6];
    real_recv_polyn = new double * [6];

    for(i=0; i<6; i++){
        real_send_polyn[i] = new double [to_send_buffer_len];
        real_recv_polyn[i] = new double [to_recv_buffer_len];
    }

}

void molecule:: free_space_for_real_Chebyshev_method(){
    delete [] real_time_normalized_mat;
    delete [] real_Bessel_function_array;

    int i;
    for(i=0;i<6;i++){
        delete [] real_send_polyn[i];
        delete [] real_recv_polyn[i];
    }
    delete [] real_send_polyn;
    delete [] real_recv_polyn;
}

void molecule:: Chebyshev_method_real_time_single_wave_func(vector<double> & wave_func_real , vector<double> & wave_func_imag
                                                            , bool forward_evolve){
    int i, j, k , m;
    int irow_index , icol_index;
    double bess;
    double air, aii;

    update_individual_wave_func(wave_func_real);
    update_individual_wave_func(wave_func_imag);

    vector<double> & creal =  wave_func_real;
    vector<double> & cimag = wave_func_imag ;

    double time_evolve_sign;
    if (forward_evolve){
        time_evolve_sign = + 1; // evolve forward
    }
    else{
        time_evolve_sign = -1; // evolve backward.
        // it is equivalent to setting delt = - delt in function prepare_Chebyshev_polynomial_evolution_real_time
        // need to take care of real_Chebyshev_R_t, real_Chebyshev_expi.
        // for real_Chebyshev_expi : real_Chebyshev_expi * (-1)
        // for real_Chebyshev_R_t : real_Bessel_function_array[i] = real_Bessel_function_array[i] * (-1)^{i}. See Bessel function of first kind :  https://www.wikiwand.com/en/Bessel_function#Bessel_functions_of_the_first_kind:_J.CE.B1
    }

    for(i=0; i< basis_set_num + to_recv_buffer_len; i++){
        // [0], [2], [4] are real part of the wave function. [1],[3],[5] are imaginary part of the wave function.
        real_time_Chebyshev_polyn[0][i] = creal[i];
        real_time_Chebyshev_polyn[1][i] = cimag[i];
        real_time_Chebyshev_polyn[2][i] = 0;
        real_time_Chebyshev_polyn[3][i] = 0;
        real_time_Chebyshev_polyn[4][i] = 0;
        real_time_Chebyshev_polyn[5][i] = 0;
    }

    // zeroth order, C0 = 1, J0(Rt) * exp(-i* e0 * dt) * wavefunction
    // air, aii represent real and imaginary part of prefactor.  See second page of my note for definition of prefactor ak.
    bess = real_Bessel_function_array[0];
    air = real_Chebyshev_expr * bess;
    aii = real_Chebyshev_expi * bess * time_evolve_sign;

    // creal, cimag = a0 * T0(omega)
    for(i=0;i< basis_set_num ;i++){
        creal[i] = air * real_time_Chebyshev_polyn[0][i] - aii * real_time_Chebyshev_polyn[1][i];
        cimag[i] = air * real_time_Chebyshev_polyn[1][i] + aii * real_time_Chebyshev_polyn[0][i];
    }

    // first order of Chebyshev polynomial. See third page of my note for recursive relation of Chebyshev polynomial.
    bess = real_Bessel_function_array[1] * pow(time_evolve_sign , 1); // J1(R)
    air = 2 * bess * real_Chebyshev_expr; // C1 = 2.  real part of prefactor
    aii = 2 * bess * real_Chebyshev_expi * time_evolve_sign; // imaginary part of prefactor.
    for(i = 0;i < mat_num; i++){
        irow_index = local_irow[i];
        icol_index = local_icol[i];
        // omega * psi, here omega = (-i) * normalized_Hamiltonian.
        // imaginary part of (-i) * (H - E) / R  * psi.
        real_time_Chebyshev_polyn[3][irow_index] = real_time_Chebyshev_polyn[3][irow_index] - real_time_normalized_mat[i] * real_time_Chebyshev_polyn[0][icol_index];
        // real part of (-i) * (H - E) / R * psi.
        real_time_Chebyshev_polyn[2][irow_index] = real_time_Chebyshev_polyn[2][irow_index] + real_time_normalized_mat[i] * real_time_Chebyshev_polyn[1][icol_index];
    }

    // creal , cimag = a0 * T0(omega) + a1 * T1(omega)
    for(i = 0; i < basis_set_num; i++){
        creal[i] = creal[i] + air * real_time_Chebyshev_polyn[2][i] - aii * real_time_Chebyshev_polyn[3][i];
        cimag[i] = cimag[i] + air * real_time_Chebyshev_polyn[3][i] + aii * real_time_Chebyshev_polyn[2][i];
    }

    // update wave function array real_time_Chebyshev_polyn[2] & real_time_Chebyshev_polyn[3] for matrix vector multiplication.
    update_polyn23(real_send_polyn, real_recv_polyn, real_time_Chebyshev_polyn);

    // Remaining terms. Use the recursive relation of the Chebyshev polynomial to compute remaining terms. See page 3 of the attached note.
    // T_{k}(omega) * psi = 2 * omega * T_{k-1}(omega) * psi + T_{k-2}(omega) * psi.
    for(k = 2; k <= real_N_chebyshev; k++){
        bess = real_Bessel_function_array[k] * pow(time_evolve_sign, k);
        air = 2 * bess * real_Chebyshev_expr; // Ck = 2. bess = bessel function of order k. air : real part of prefactor
        aii = 2 * bess * real_Chebyshev_expi * time_evolve_sign; // aii : imaginary part of pre-factor.

        // use Chebychev polynomial recursion relationship. J_{k+2}( -i *  normalized_wave_func) = J_{k+1}(-i * normalized_wave_func) *2 * (-i  *normalized_wave_func) + J_{k}( -i * normalized_wave_func), normalized_wave_func=  (H - E)/ R
        for(i=0; i< mat_num; i++) {
            irow_index = local_irow[i];
            icol_index = local_icol[i];
            // 2 * (-i (H - E) / R) * T_{k-1} (omega) * psi. [5] : imaginary part of T_{k}, [2] : real part of T_{k-1}
            real_time_Chebyshev_polyn[5][irow_index] = real_time_Chebyshev_polyn[5][irow_index]
                                                       - 2 * real_time_normalized_mat[i] *
                                                         real_time_Chebyshev_polyn[2][icol_index];
            // 2 * (-i (H - E) / R) * T_{k-1} (omega) * psi. [4] : real part of T_{k}. [3]: imaginary part of T_{k-1}
            real_time_Chebyshev_polyn[4][irow_index] = real_time_Chebyshev_polyn[4][irow_index]
                                                       + 2 * real_time_normalized_mat[i] *
                                                         real_time_Chebyshev_polyn[3][icol_index];
        }

        for(i = 0; i < basis_set_num; i++){
            //  + T_{k-2}(omega) * psi.
            real_time_Chebyshev_polyn[4][i] = real_time_Chebyshev_polyn[4][i] + real_time_Chebyshev_polyn[0][i];
            real_time_Chebyshev_polyn[5][i] = real_time_Chebyshev_polyn[5][i] + real_time_Chebyshev_polyn[1][i];
        }

        // update wave function.
        // + a_{k} * T_{k} (omega)
        for(i=0;i<basis_set_num; i++){
            creal[i] = creal[i] + air * real_time_Chebyshev_polyn[4][i] - aii * real_time_Chebyshev_polyn[5][i];
            cimag[i] = cimag[i] + air * real_time_Chebyshev_polyn[5][i] + aii * real_time_Chebyshev_polyn[4][i];
        }

        // to compute next order of Chebyshev polynomial, we make T_{k} -> T_{k-1}.
        for(i = 0;i < basis_set_num; i++){
            real_time_Chebyshev_polyn[0][i] = real_time_Chebyshev_polyn[2][i];
            real_time_Chebyshev_polyn[1][i] = real_time_Chebyshev_polyn[3][i];
            real_time_Chebyshev_polyn[2][i] = real_time_Chebyshev_polyn[4][i];
            real_time_Chebyshev_polyn[3][i] = real_time_Chebyshev_polyn[5][i];
            real_time_Chebyshev_polyn[4][i] = 0;
            real_time_Chebyshev_polyn[5][i] = 0;
        }
        update_polyn23(real_send_polyn, real_recv_polyn, real_time_Chebyshev_polyn);
    }

}

void molecule:: Chebyshev_method_real_time_multiple_wave_func(vector<vector<double>> & wave_func_real,
                                                              vector<vector<double>> & wave_func_imag, double real_time,
                                                              bool forward_evolve){
    int i,j;
    int step_number;
    step_number = int(real_time / delt);

    // Evolve multiple wave function forward in time by treating them individually.
    for(i=0;i<Haar_random_state_num;i++){
        for(j=0;j<step_number;j++){
            Chebyshev_method_real_time_single_wave_func(wave_func_real[i], wave_func_imag[i] , forward_evolve);
        }
    }


}