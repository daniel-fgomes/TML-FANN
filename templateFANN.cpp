/* 
 * File:   main.cpp
 * Author: Daniel de Filgueiras Gomes ( daniel.fgomes@ufpe.br )
 * This source code is licensed under the MLP2 terms (MLP2.html or https://www.mozilla.org/en-US/MPL/2.0/ )
 * This code is provided as an illustrative example in Machine Learning course in Department of Electronics and Systems/UFPE.
 * ( https://www.ufpe.br/des/o-des )
 * Created on 17 de Agosto de 2018, 11:00
 */

#include <stdio.h>
#include <fann.h>
#include "floatfann.h"

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Important warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//This program aims to solve the XOR classification problem.
//This is an toy problem and for simplicity the cross-validation procedures and parameter settings are not treated here.

int execution()
{
    fann_type *calc_out;
    fann_type input[2];

    struct fann *ann = fann_create_from_file("xor_float.net");

    input[0] = -1;
    input[1] = 1;
    calc_out = fann_run(ann, input);

    printf("xor test (%f,%f) -> %f\n", input[0], input[1], calc_out[0]);

    fann_destroy(ann);
    return 0;
}

int training() {
    const unsigned int num_input = 2;
    const unsigned int num_output = 1;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 3;
    const float desired_error = (const float) 0.001;
    const unsigned int max_epochs = 500000;
    const unsigned int epochs_between_reports = 1000;

    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);

    fann_save(ann, "xor_float.net");

    fann_destroy(ann);

    return 0;
}

int main() {
    execution();
    
    return 0;
}