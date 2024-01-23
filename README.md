# Note about Chebyshev method code

The code is based on Chebyshev method for evolving wave function, which is documented in this paper: [An accurate and efficient scheme for propagating the time dependent Schrödinger equation | The Journal of Chemical Physics | AIP Publishing](https://aip.scitation.org/doi/10.1063/1.448136)

In the ./reference/ folder, I also provide one pdf slide which briefly discusses this method.

In the code/ folder, I attach the code that evolve wave function using Chebyshev method. This code is a parallel code which uses MPI for parallel computing. 

In this note, I will explain the attached code and how it implements Chebyshev method.

## Code structure

#### 1. prepare_evolution.cpp

This function is used to enable multiple processes to work together using MPI for matrix vector multiplication.

This code is needed for implementing any multi-process matrix-vector multiplication (Of course, the way I write it), not specific for Chebyshev method. (You will likely see the same code for parallel SUR algorithm I have written)

See figures below for explanation of  the idea:

![](/home/phyzch/CLionProjects/Chebyshev%20method/note%20fig/PNG%20image.png)

![](/home/phyzch/CLionProjects/Chebyshev%20method/note%20fig/PNG%20image1.png)

We use MPI to speed up matrix array multiplication.

We divide wave function $\psi$ into different parts and store it in different processes.

$$
d\psi = \psi(t+dt) - \psi(t) = dt \times (H \psi)
$$

Hamiltonian matrix H (sparse matrix) elements are stored in (matrix element value, row index, column index) format (Dictionary of keys). These pairs are shared between different processes (See cartoon above).

$~$

We know Hamiltonian H as a sparse matrix may have non-zero component : $H(m,n)\neq 0$ , and index $m$ , $n$ belong to different processes.

$~$

To be more specific:

For process $p_n$, it has an array of wave function $\psi$ from index $n_{1} \sim n_{N}$ , while components of wave function with index other than $n_{1} \sim n_{N}$ are stored in other processes. We could have nonzero matrix elements H($n_{j}$,m) where $n_{j} \in [n_{1} , n_{N}]$ but $m \not \in [n_{1} , n_{N}]$ . Therefore, to compute wave function component $\psi(n_{j}, t+ \Delta t)$ at next time step $t+\Delta t$, we need wave function component  value $\psi(m, t)$ at time $t$. The wave function component $\psi(m,t)$ is stored in other processes. We need to use MPI to send $\psi(m,t)$ to process $p_{n}$ every time we evolve wave function one step forward. For the time-independent Hamiltonian $H$ we use here, the positions of wave function component $\psi(m)$ and $\psi(n)$ are fixed, therefore, we can figure out the information of wave function components' position for MPI before we evolve the wave function. 

$~$

After we figure out that there are in total M elements that correspond to $H(n_{j},m)\neq 0$  and $\psi(m)$ not in the process $p_{j}$, then we extend the wave function $\psi$ from the size $n_{N} - n_{1}$ to $n_{N} - n_{1} + M$. The extra $M$ components at the end of the wave function array in each process will store wave function components $\psi(m)$ it receives from other processes.

$~$

During wave function propagation, each time before we compute $H \psi$ , we send data $\psi[m]$ from other processes and store them in extra M elements at the end of wave function array $\psi$. Then when compute  wave function at next time step $\psi(n_{j}, t + \Delta t ) = \psi(n_{j} , t) + \sum_{m} H(n_{j},m) \psi(m,t) \times \Delta t $ ,

for part $\psi(m,t)$ belongs to other processes, we multiply $H(n_{j}, m)$ by elements at the end of the wave function array.

To better understand the function **prepare_evolution()** , we need to understand which MPI function we use to communicate wave function components between processes. The MPI function we use is [MPI_Alltoallv()](https://www.mpich.org/static/docs/v3.0.x/www3/MPI_Alltoallv.html). To use MPI_Alltoallv(), as the hyperlink has suggested, we need sendbuf (address of send array in each processes), sendcounts (the size of array to send in each processes), sdispls (the displacement of send buffer in each processes), sendtype (type of variables to send, INT, DOUBLE or BOOL). Similarly, we have recvbuf, recvcounts, rdispls, recvtype for process to receive data. 

To understand the functioin MPI_Alltoallv(), it helps to first understand the function MPI_Alltoall(), as they are very similar. 

A great visualization of MPI_alltoall function is shown in picture below:

![](/home/phyzch/CLionProjects/Chebyshev%20method/note%20fig/MPI_alltoall.png)

Here, we can see each process receive parts of data from other processes. The function MPI_alltoallv () is a variation of MPI_alltoall(). Using MPI_alltoallv(), we can send and receive data with different lengths, instead of data of the same length as in MPI_ALltoall(). 

The way we communicate the wave function components $\psi(m)$ using MPI_Alltoallv() is following:

(1) we figure out the wave function components we need to send to other processes, which we record in **tosendVecIndex** .

(2) We store the value of wave function components in send_buffer, using the **tosendVecIndex**

(3) We use MPI_Alltoallv() function to send data in send_buffer to recv_buffer. To use MPI_Alltoallv(), as indicated above, we need sendcounts (tosendVecCount), sdispls (tosendVecPtr), recvcounts (remoteVecCount), rdispls (remoteVecPtr). These are all arrays with length (num_proc), and they are computed in the **prepare_evolution()**

function.

(4) The recv_buffer load the received data into extra space at the end of the wave function array $\psi$ we prepared for receiving data. 

In matrix vector multiplication, the remoteVecIndex records the index of the received data at the end of wave function. 

Explanation of the function:

##### 1.1 construct_receive_buffer_index(remoteVecCount, remoteVecPtr, remoteVecIndex)

Figure out four variables: (1) to_recv_buffer_len (total received buffer length from other processes) 

(2) remoteVecCount (number of wave function elements to receive from each processes)

(3) remoteVecPtr (displacement of wave function buffers we received)

(4) remoteVecIndex (index of wave function components from other processes)

The code should be self-explained.

#### 1.2 construct_send_buffer_index

This function figures out which wave function vector elements to send to other processes. 

It uses the variable remoteVecCount, remoteVecPtr, remoteVecIndex we computed in function **construct_receive_buffer_index**

It outputs the information about sending out wave function data to other processes:

**tosendVecCount_element** , **tosendVecPtr_element**, **tosendVecIndex_Ptr**.

We use MPI_Alltoall() function to get information about number of elements to send to other processes.

We use MPI_Alltoallv() function to get information about global Index of wave function vector to send to other processes.

#### 1.3 prepare_evolution()

This function first call construct_recv_buffer_index() to figure out wave function vector index to receive from other processes. Then call construct_send_buffer_index() to figure out the wave function vector index to send to other processes.

Then it will compute **local_irow** and **local_icol**. These will make matrix vector multiplication performed in each process as if all elements are local in process. The way it functions is following:

(1) local_irow[i] = irow[i]  - my_id * vsize:  As name local_irow has indicated, local_irow is the local index in the process for row index of sparse matrix.

(2) local_icol[i]: This is column index for matrix multiplication. As we have indicated above, the elements received from other processes are stored at the end of the wave function array. 

Therefore,  local_icol[i] will be the local index in the wave function array if this element is already in the process. 

```cpp
 else{ // this column index is in local matrix.
       local_icol.push_back (icol[i] - my_id * vsize );
     }
```

Otherwise, it will be the index of the one we attached to the end of the wave function array:

```cpp
if(search_Ind!=NULL){
   // this column index is not in local wave function array, and we should get it from other process (remoteVec)
   local_icol.push_back(basis_set_num + (search_Ind - remoteVecIndex) );
   }
```



### 2. Evolve_Chebyshev_method_real_time

This cpp file include functions to evolve the wave function $\psi$ in real time with Chebyshev polynomial. For the explanation of the Chebyshev method, see the note in the ./reference/Numerical method Chebyshev.pdf and also the original paper.



The function in the **Evolve_Chebyshev_method_real_time.cpp** should be self-explained with the comment I provided. I will provide detailed explanation in this note below for this code.



#### 2.1 prepare_Chebyshev_polynomial_evolution_real_time

This function prepare variables we used to evolve wave function in real time using Chebyshev polynomials.



Explanation of the variables:

(1) **real_Chebyshev_R** = $(V_{max} - V_{min}) * 0.55$

(2) **real_Chebyshev_e0** =  $E$ = ($V_{max} + V_{min}$ ) / 2 . mean of Hamiltonian mean value by estimate.

(3) **real_time_normalized_mat** = $(H - E)/R$ :  normalized Hamiltonian.

(4) **real_Chebyshev_R_t_** = $(V_{max} - V_{min}) * 0.55 * dt$

(5) **real_Chebyshev_expr** , **real_Chebyshev_expi** : real and imaginary part of prefactor $e^{i(E*dt)}$ . real part = $\cos(E * dt)$ , imag part = $\sin(E * dt)$  

(6) **N_Chebyshev** : order of Chebyshev polynomial to use.

(7) **real_Bessel_function_array** : compute bessel function $J_{k}(R*dt)$, 

(8) **real_time_Chebyshev_polyn** : this is used to store Chebyshev polynomial $T_{k}(-iHdt) \psi $

(9) **real_send_polyn** , **real_recv_polyn** :  Used for MPI to communicate about the Chebyshev polynomial when we do matrix vector multiplication across different processes. Used in function: **update_poly23()**.  **update_poly23()** function is used both for evolving wave function in real time and imaginary time.



#### 2.2 Chebyshev_method_real_time_single_wave_func()

Evolve a wave function forward in time $dt$ , 

$$
\psi(t+dt) = e^{-i\hat{H} dt} \psi(t) = \sum_{k=  0}^N a_k T_k(\hat{\omega}) \psi(t)
$$

Here $a_{n}$ is prefactor, $T_{n}$ is the Chebyshev polynomial. $\hat{\omega} = - i (H - E) / R $,  See previous section for $E$ and $R$ .The details are given as following:

$$
a_{k} = e^{i E \cdot dt} \times C_{k} \times J_{k}(R dt)
$$

$E = (V_{max} + V_{min})/2$

Here $C_{k} = 1$ if $k=0$,  $C_{k} = 2$ if $k \geq 0$ . 

$J_{k}(R)$ is Bessel function of first kind of order $k$, where $R = (V_{max} - V_{min}) * 0.55$



$T_{k}$ are Chebyshev polynomial, in the program, they are computed using the recursive relation as following:

$T_{0}(\hat{\omega}) = 1$ , $T_{1}(\hat{\omega}) = \hat{\omega}$ .

which implies,

$T_{0}(\hat{\omega}) \psi = \psi$ , $T_{1} (\hat{\omega}) \psi = \hat{\omega} \times \psi$

$$
T_k(\widehat{\omega}) \psi=2 \widehat{\omega}\left(T_{k-1}(\widehat{\omega}) \psi\right)+T_{k-2}(\widehat{\omega}) \psi
$$



Below we explain the code in detail:

(1)

```cpp
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
```

**bess** = $J_{0}(R \cdot dt)$ , **air** = $\cos(E \cdot dt) \times J_{0}(R \cdot dt)$ . **aii** = $\sin(E \cdot dt) \times J_{0}(R \cdot dt)$

At this stage, **creal , cimag**  are real and imag parts of: $a_{0} T_{0}(\hat{\omega}) \psi(t) = a_{0} \psi(t)$ , 

 

(2)

```cpp
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
        // real part of (-i) * (H - E) / R  * psi.
        real_time_Chebyshev_polyn[2][irow_index] = real_time_Chebyshev_polyn[2][irow_index] + real_time_normalized_mat[i] * real_time_Chebyshev_polyn[1][icol_index];
    }

    // creal , cimag = a0 * T0(omega) + a1 * T1(omega)
    for(i = 0; i < basis_set_num; i++){
        creal[i] = creal[i] + air * real_time_Chebyshev_polyn[2][i] - aii * real_time_Chebyshev_polyn[3][i];
        cimag[i] = cimag[i] + air * real_time_Chebyshev_polyn[3][i] + aii * real_time_Chebyshev_polyn[2][i];
    }
```

**bess** = $J_{1}(R \cdot dt)$ , **air** = $2 \cos(E \cdot dt) \times J_{1}(R \cdot dt)$ , **aii** = $2 \sin(E \cdot dt) \times J_{1}(R \cdot dt)$ 



**real_time_Chebyshev_polyn[2]** : real part of Chebyshev polynomial. $T_{1}(\hat{\omega}) \psi$

**real_time_Chebyshev_polyn[3]** : imaginary part of Chebyshev polynomial. $T_{1}(\hat{\omega}) \psi$



$T_{1}(\hat{\omega}) \psi = \hat{\omega} \psi$ , here $\hat{\omega} = (\hat{H} - E) / R$

At this stage, **creal** , **cimag** are real and imag parts of $a_{0} T_{0}(\hat{\omega}) \psi(t) + a_{1} T_{1}(\hat{\omega}) \psi(t)$.



(3)

```cpp
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
```

For the matrix vector multiplication, we need to update wave function array before we do it, as this matrix vector multiplication is implemented across different processes.



The following code can be summarized as following:

**real_time_Chebyshev_polyn[4]** : real part of $T_{k}(\hat{\omega})$ 

**real_time_Chebyshev_polyn[5]** : imag part of $T_{k}(\hat{\omega})$

**real_time_Chebyshev_polyn[2]** : real part of $T_{k-1}(\hat{\omega})$

**real_time_Chebyshev_polyn[3]** : imag part of $T_{k-1}(\hat{\omega})$

**real_time_Chebyshev_polyn[0]** : real part of $T_{k-2}(\hat{\omega})$

**real_time_Chebyshev_polyn[1]** : imag part of $T_{k-2}(\hat{\omega})$



WIth the equation:

$$
T_k(\widehat{\omega}) \psi=2 \widehat{\omega}\left(T_{k-1}(\widehat{\omega}) \psi\right)+T_{k-2}(\widehat{\omega}) \psi
$$

We can see when we compute $T_{k}(\hat{\omega}) \psi$  (**real_time_Chebyshev_polyn[4]** , **real_time_Chebyshev_polyn[5]**) , we only need matrix vector multiplication for term: $\hat{\omega} \cdot T_{k-1}(\hat{\omega}) \psi$ , therefore, only $T_{k-1}(\hat{\omega}) \psi$ (**real_time_Chebyshev_polyn[2]**, **real_time_Chebyshev_polyn[3]**) need to be updated at each time step. This is performed by using function **update_poly23()**.



Other parts of the code should be self-explanary.



#### update_poly23()

As we have mentioned in the previous section, **update_poly23()** is used to update **real_time_Chebyshev_polyn[2]**, **real_time_Chebyshev_polyn[3]** .

The pre-requisite for using this function is by calling **prepare_evolution()** function in the **prepare_evolution.cpp** , which will prepare the variable for communication of wave function vectors between processes.



Let's look at this code in detail, readers should also refer to **prepare_evolution()** function for better understanding this part.

(1)  Collect the wave function component need to be sent:

```cpp
    vsize = total_basis_set_num /num_proc;
    begin_index = vsize * my_id;
    for(i = 0;i < to_send_buffer_len; i++){
        send_polyn[2][i] = Chebyshev_polyn[2][tosendVecIndex[i] - begin_index];
        send_polyn[3][i] = Chebyshev_polyn[3][tosendVecIndex[i] - begin_index];
    }
```

**tosendVecIndex** records the index of wave function components need to be sent to other processes. As this index is index across different processes (global index), we can get local index in the local array (**Chebyshev_polyn**) by substract it with the offset : **begin_index**.

The wave function data we need to send to other processes are collected in **send_polyn[2]** , **send_polyn[3]**.



(2)

```cpp
MPI_Alltoallv(&send_polyn[2][0],tosendVecCount,tosendVecPtr,MPI_DOUBLE,
                  &recv_polyn[2][0],remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);
MPI_Alltoallv(&send_polyn[3][0],tosendVecCount,tosendVecPtr,MPI_DOUBLE,
                  &recv_polyn[3][0],remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);
```

Use [**MPI_Alltoallv()**](https://www.mpich.org/static/docs/v3.0.x/www3/MPI_Alltoallv.html) function to send the wave function data in **send_polyn** to **recv_polyn**. The information for sending these data (**tosendVecCount**, **tosendVecPtr**) and receiving these data (**remoteVecCount** , **remoteVecPtr**) are constructed in **preprare_evolution()** function.



(3)

```cpp
     for(i = 0;i < to_recv_buffer_len; i++){
        Chebyshev_polyn[2][ i + basis_set_num ]  = recv_polyn[2][i];
        Chebyshev_polyn[3][ i + basis_set_num ]  = recv_polyn[3][i];
    }
```

Append the wave function components received from other processes at the end of the wave function array. 

Remember, the **local_icol** will point to these data when we implemenet matrix vector multiplication.



(4) After using **update_poly23()** function, the matrix vector multiplication for updating wave function is performed as following:

```cpp
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
```



We can see now we use **local_irow** and **local_icol** to perform matrix vector multiplication across different processes. **local_icol** will point to the additional elements at the end of wave function array if they it correpond to the data received from other processes. See **prepare_evolution()** function for more detail.






