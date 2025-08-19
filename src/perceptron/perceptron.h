#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "dataset.h"

// Struttura del percettrone
typedef struct {
    double *weights;     // Pesi sinaptici
    double bias;         // Bias
    int num_inputs;      // Numero di input
    double learning_rate; // Tasso di apprendimento
} Perceptron;

// Funzioni principali
Perceptron* perceptron_create(int num_inputs, double learning_rate);
void perceptron_destroy(Perceptron *p);
void perceptron_initialize_weights(Perceptron *p);
double perceptron_activate(double sum);
int perceptron_predict(Perceptron *p, double *inputs);
double perceptron_compute_output(Perceptron *p, double *inputs);
void perceptron_train_single(Perceptron *p, double *inputs, int target);
int perceptron_train_dataset(Perceptron *p, Dataset *data, int max_epochs);
void perceptron_print_weights(Perceptron *p);

#endif
