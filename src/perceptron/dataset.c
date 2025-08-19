#include "dataset.h"

// Crea un nuovo dataset
Dataset* dataset_create(int num_samples, int num_features) {
    Dataset *data = malloc(sizeof(Dataset));
    if (!data) return NULL;
    
    data->inputs = malloc(num_samples * sizeof(double*));
    data->targets = malloc(num_samples * sizeof(int));
    
    if (!data->inputs || !data->targets) {
        free(data);
        return NULL;
    }
    
    for (int i = 0; i < num_samples; i++) {
        data->inputs[i] = malloc(num_features * sizeof(double));
        if (!data->inputs[i]) {
            // Cleanup in caso di errore
            for (int j = 0; j < i; j++) {
                free(data->inputs[j]);
            }
            free(data->inputs);
            free(data->targets);
            free(data);
            return NULL;
        }
    }
    
    data->num_samples = num_samples;
    data->num_features = num_features;
    return data;
}

// Distrugge il dataset
void dataset_destroy(Dataset *data) {
    if (data) {
        if (data->inputs) {
            for (int i = 0; i < data->num_samples; i++) {
                if (data->inputs[i]) free(data->inputs[i]);
            }
            free(data->inputs);
        }
        if (data->targets) free(data->targets);
        free(data);
    }
}

// Aggiunge un campione al dataset
void dataset_add_sample(Dataset *data, int index, double *inputs, int target) {
    if (index >= 0 && index < data->num_samples) {
        for (int i = 0; i < data->num_features; i++) {
            data->inputs[index][i] = inputs[i];
        }
        data->targets[index] = target;
    }
}

// Crea dataset per la funzione AND
Dataset* dataset_create_and(void) {
    Dataset *data = dataset_create(4, 2);
    if (!data) return NULL;
    
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int targets[4] = {0, 0, 0, 1};
    
    for (int i = 0; i < 4; i++) {
        dataset_add_sample(data, i, inputs[i], targets[i]);
    }
    
    return data;
}

// Crea dataset per la funzione OR
Dataset* dataset_create_or(void) {
    Dataset *data = dataset_create(4, 2);
    if (!data) return NULL;
    
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int targets[4] = {0, 1, 1, 1};
    
    for (int i = 0; i < 4; i++) {
        dataset_add_sample(data, i, inputs[i], targets[i]);
    }
    
    return data;
}

// Crea dataset per la funzione XOR (non linearmente separabile!)
Dataset* dataset_create_xor(void) {
    Dataset *data = dataset_create(4, 2);
    if (!data) return NULL;
    
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int targets[4] = {0, 1, 1, 0};
    
    for (int i = 0; i < 4; i++) {
        dataset_add_sample(data, i, inputs[i], targets[i]);
    }
    
    return data;
}
