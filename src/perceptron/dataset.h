#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Struttura per i dataset di training
typedef struct {
    double **inputs;     // Matrice degli input
    int *targets;        // Valori target (0 o 1)
    int num_samples;     // Numero di campioni
    int num_features;    // Numero di feature per campione
} Dataset;

// Funzioni principali
Dataset* dataset_create(int num_samples, int num_features);
void dataset_destroy(Dataset *data);
void dataset_add_sample(Dataset *data, int index, double *inputs, int target);
Dataset* dataset_create_xor(void);
Dataset* dataset_create_and(void);
Dataset* dataset_create_or(void);

#endif
