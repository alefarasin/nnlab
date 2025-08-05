#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Struttura del percettrone
typedef struct {
    double *weights;     // Pesi sinaptici
    double bias;         // Bias
    int num_inputs;      // Numero di input
    double learning_rate; // Tasso di apprendimento
} Perceptron;

// Struttura per i dataset di training
typedef struct {
    double **inputs;     // Matrice degli input
    int *targets;        // Valori target (0 o 1)
    int num_samples;     // Numero di campioni
    int num_features;    // Numero di feature per campione
} Dataset;

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

// Funzioni per gestire i dataset
Dataset* dataset_create(int num_samples, int num_features);
void dataset_destroy(Dataset *data);
void dataset_add_sample(Dataset *data, int index, double *inputs, int target);
Dataset* dataset_create_xor(void);
Dataset* dataset_create_and(void);
Dataset* dataset_create_or(void);

#endif
