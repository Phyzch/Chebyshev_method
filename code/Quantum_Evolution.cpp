//
// Created by phyzch on 1/23/24.
//
#include "util.h"
#include "molecule.h"
using namespace std;

void molecule::Quantum_Evolution(){
    double t;
    int i,j,k;
    int print_data_num = int(tmax / tprint) + 1;
    bool forward_evolve_bool = true; // true if evolve wave function forward in time. false if evolve backward.
    vector<double> time_list;
    double imag_time_beta = 5;
    vector<vector<double>> imag_time_evolved_multiple_wave_func_real, imag_time_evolved_multiple_wave_func_imag;

    // fixme : initialize Hamltonian

    // prepare send and receive buffer for MPI state evolution.
    prepare_evolution();
    // prepare wave function evolution using Chebyshev method in imag time;
    prepare_Chebyshev_polynomial_evolution_imag_time();
    // prepare wave function using Chebyshev method in real time;
    prepare_Chebyshev_polynomial_evolution_real_time();

    // Evolve wave function
    // fixme: initialize wave function psi(0) for evolution

    t = 0;
    for( i = 0; i < print_data_num; i++){
        //fixme: calculate something with wave function

        // if evolve wave function in real time.
        Chebyshev_method_real_time_multiple_wave_func(multiple_wave_func_real, multiple_wave_func_imag,
                                                      tprint, forward_evolve_bool);
    }

    // or below we can evolve wave function in imaginary time
    Chebyshev_method_imag_time_multiple_wave_func(multiple_wave_func_real, multiple_wave_func_imag,
                                                  imag_time_evolved_multiple_wave_func_real, imag_time_evolved_multiple_wave_func_imag,
                                                  imag_time_beta);

}