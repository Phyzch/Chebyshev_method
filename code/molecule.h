//
// Created by phyzch on 12/28/22.
//
#pragma once
#include<iostream>
#include<iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include<ctime>
using namespace std;

class molecule{
    // first axis is for reactive coordinate, which is on coordinate basis set.
    // second axis is for harmonic bath mode, which is on harmonic basis set

private:
    vector <int> irow, icol; // row and column index for sparse matrix element
    vector<double> mat; // Hamiltonian matrix
    vector<double> total_mat; // Hamiltonian matrix across differnt process
    vector<int> total_irow, total_icol;

    int mat_num;
    int total_mat_num;
    vector<int> matnum_each_process;

    vector<vector<double>> multiple_wave_func_real, multiple_wave_func_imag;  // real part of wave function multiple_wave_func_real, imaginary part of wave function multiple_wave_func_imag
    vector<vector<double>> multiple_wave_func_real_all, multiple_wave_func_imag_all;

    int * remoteVecCount,  * remoteVecPtr,  *  remoteVecIndex;
    int * tosendVecCount,  *tosendVecPtr,  * tosendVecIndex;
    int  to_send_buffer_len,  to_recv_buffer_len;
    double ** recv_wave_func_real,  ** recv_wave_func_imag,  ** send_wave_func_real,  ** send_wave_func_imag;
    vector <int >  local_irow;
    vector <int >  local_icol;

    vector< vector<int> > bath_state_qn_all ; // record bath mode for states.
    vector<double> bath_state_energy_all; // record energy for bath state.
    int bath_state_num;
    double bath_state_energy_cutoff; // energy window for including states (in boson system, we have infinite number of states)

    vector<double>  basis_set_energy_all; // matrix in all process
    vector<double> basis_set_potential_energy_all;
    vector<int> basis_set_coordinate_eigenstate_index_all; // record coordinate index for all states.
    vector<int> basis_set_bath_state_index_all; // record bath state index for all states.
    int total_basis_set_num;
    int basis_set_num;
    int * basis_set_num_each_proc;
    vector<int> basis_set_num_each_proc_array;
    int * basis_set_num_offset_each_proc;

    double delt, tmax, tprint; // delt: time step for propagation the wave function.
                               // tmax: maximum time for wave function propagation.
                               // tprint : time to print out put.

    double mass;
    double anharmonic_V3, anharmonic_scaling_factor; // V3: anharmonic cubic coupling strength. anharmonic_scaling_factor is the scaling factor for higher order anharmonic coupling.
    double anharmonic_V0;  // prefactor for anharmonic coupling.
    double anharmonic_coupling_cutoff;
    vector<double> scaling_factor_for_mode;

    vector<vector<int>> anharmonic_coupling_bath_state_index_list;
    vector<vector<double>> anharmonic_coupling_strength_list;

    vector<double> reaction_pot_param; // parameter for reaction potential
    int reaction_param_num;

    int nmode; // number of bath mode
    double root_mean_square_frequency;
    vector<double> mfreq; // frequency of bath modes
    vector<double> bath_linear_coupling_strength; // ci q xi , here ci is linear coupling strength
    vector<int> nmax; // maximum quantum number for bath modes.

    int Haar_random_state_num;

    vector<double> temperature_list;

    string path;
    // output and input of file
    ofstream output; // output result we record.
    ofstream log;  // output status (problem with code)
    ifstream input;

    // Chebyshev time evolution
    double  imag_Chebyshev_e0, imag_Chebyshev_R, imag_Chebyshev_prefactor, imag_Chebyshev_R_times_imag_time;
    double *  imag_time_shifted_mat;
    int imag_N_chebyshev;
    double *  imag_Bessel_function_array;
    vector<vector<double>>   imag_time_Chebyshev_polyn;
    double **  imag_send_polyn, ** imag_recv_polyn;

    // for real time Chebyshev method. (to avoid bug and possible confliction, I declare another set of parameters)
    double real_Chebyshev_e0, real_Chebyshev_R, real_Chebyshev_exp_real, real_Chebyshev_exp_imag , real_Chebyshev_R_t;
    double * real_time_normalized_mat;
    int real_N_chebyshev;
    double * real_Bessel_function_array;
    vector<vector<double>> real_time_Chebyshev_polyn;
    double ** real_send_polyn, ** real_recv_polyn;

    // x_evl: quadrature points for DVR, x_eig_vector: eigenvector for transformation between DVR and FBR. we choose hermitian basis set.
    vector<double> x_evl;
    vector<vector<double>> x_eig_vector;
    vector<vector<double>> coordinate_first_order_derivative_DVR;
    vector<vector<double>> coordinate_operator_second_order_derivative_DVR;

    // eigenbasis along reaction coordinate for proton transfer reaction
    int reaction_coordinate_evl_basis_num;
    vector<vector<double>> reaction_coordinate_Hamiltonian_eigenstate;
    vector<double> reaction_coordinate_Hamiltonian_eigenvalue;
    vector<vector<double>> q_eigenstate_representation;
    vector<vector<double>> projection_operator_eigenstate_representation;

    vector<vector<double>> first_order_derivative_eigenstate_representation; // <evt | d/dx |evt>


public:
    void Quantum_Evolution();

    // prepare evolution
    void prepare_evolution();
    int construct_receive_buffer_index(int * remoteVecCount_element, int * remoteVecPtr_element, int * remoteVecIndex_element);


    void update_real_part(vector<vector<double>> & wave_func_real);
    void update_imag_part(vector<vector<double>> & wave_func_imag);
    void update_individual_wave_func(vector<double> & wave_func);

    // evolve wave function imaginary time using Chebyshev method
    void prepare_Chebyshev_polynomial_evolution_imag_time( );

    void update_polyn23( double ** & send_polyn, double ** & recv_polyn, vector<vector<double>> &  Chebyshev_polyn);

    // Chebyshev method in imaginary time
    void free_space_for_imag_Chebyshev_method();

    void Chebyshev_method_imag_time_single_wavefunc(vector<double> & wave_func_real , vector<double> & wave_func_imag);

    void Chebyshev_method_imag_time_multiple_wave_func(const vector<vector<double>> & wave_func_real , const vector<vector<double>> & wave_func_imag,
                                                       vector<vector<double>> & imag_time_evolved_wave_func_real, vector<vector<double>> & imag_time_evolved_wave_func_imag , double imag_time);

    // Chebyshev method in real time
    void prepare_Chebyshev_polynomial_evolution_real_time();
    void free_space_for_real_Chebyshev_method();
    void Chebyshev_method_real_time_single_wave_func(vector<double> & wave_func_real , vector<double> & wave_func_imag,
                                                     bool forward_evolve);
    void Chebyshev_method_real_time_multiple_wave_func(vector<vector<double>> & wave_func_real,
                                                       vector<vector<double>> & wave_func_imag, double real_time,
                                                       bool forward_evolve);





};
