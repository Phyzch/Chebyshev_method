//
// Created by phyzch on 12/28/22.
//
#pragma once
#ifndef REACTIVE_DYNAMICS_MULTI_DIMENSIONAL_SYSTEM_UTIL_H
#define REACTIVE_DYNAMICS_MULTI_DIMENSIONAL_SYSTEM_UTIL_H

#include<cmath>
#include<iostream>
#include<time.h>
#include<stdio.h>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <experimental/filesystem>
#include <random>
#include<iomanip>
#include <complex>
#include <assert.h>
#include <vector>
#include<ctime>
#include<algorithm>
#include<stdlib.h>
#include<mpi/mpi.h>
#include<sys/resource.h>
#include<list>
#include "mkl.h"

using namespace std;
#define pi2 3.141592653589793*2
#define pi 3.141592653589793

extern int my_id;
extern int num_proc;



#endif //REACTIVE_DYNAMICS_MULTI_DIMENSIONAL_SYSTEM_UTIL_H
