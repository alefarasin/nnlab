#include "perceptron.h"

// Crea un nuovo percettrone
Perceptron* perceptron_create(int num_inputs, double learning_rate) {
    Perceptron *p = malloc(sizeof(Perceptron));
    if (!p) return NULL;
    
    p->weights = malloc(num_inputs * sizeof(double));
    if (!p->weights) {
        free(p);
        return NULL;
    }
    
    p->num_inputs = num_inputs;
    p->learning_rate = learning_rate;
    p->bias = 0.0;
    
    perceptron_initialize_weights(p);
    return p;
}

// Distrugge il percettrone e libera la memoria
void perceptron_destroy(Perceptron *p) {
    if (p) {
        if (p->weights) free(p->weights);
        free(p);
    }
}

// Inizializza i pesi con valori casuali piccoli
void perceptron_initialize_weights(Perceptron *p) {
    srand(time(NULL));
    
    for (int i = 0; i < p->num_inputs; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
    }
    p->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Funzione di attivazione (gradino unitario)
double perceptron_activate(double sum) {
    return sum >= 0.0 ? 1.0 : 0.0;
}

// Predice l'output per un dato input
int perceptron_predict(Perceptron *p, double *inputs) {
    double sum = p->bias;
    
    for (int i = 0; i < p->num_inputs; i++) {
        sum += p->weights[i] * inputs[i];
    }
    
    return (int)perceptron_activate(sum);
}

// Calcola l'output continuo (prima dell'attivazione)
double perceptron_compute_output(Perceptron *p, double *inputs) {
    double sum = p->bias;
    
    for (int i = 0; i < p->num_inputs; i++) {
        sum += p->weights[i] * inputs[i];
    }
    
    return sum;
}

// Training su un singolo campione
void perceptron_train_single(Perceptron *p, double *inputs, int target) {
    int prediction = perceptron_predict(p, inputs);
    int error = target - prediction;
    
    // Aggiorna i pesi se c'è un errore
    if (error != 0) {
        for (int i = 0; i < p->num_inputs; i++) {
            p->weights[i] += p->learning_rate * error * inputs[i];
        }
        p->bias += p->learning_rate * error;
    }
}

// Training su un intero dataset
int perceptron_train_dataset(Perceptron *p, Dataset *data, int max_epochs) {
    int epoch;
    int total_errors;
    
    printf("Inizio training...\n");
    
    for (epoch = 0; epoch < max_epochs; epoch++) {
        total_errors = 0;
        
        // Passa attraverso tutti i campioni
        for (int i = 0; i < data->num_samples; i++) {
            int prediction = perceptron_predict(p, data->inputs[i]);
            if (prediction != data->targets[i]) {
                total_errors++;
                perceptron_train_single(p, data->inputs[i], data->targets[i]);
            }
        }
        
        printf("Epoca %d: %d errori\n", epoch + 1, total_errors);
        
        // Se non ci sono errori, il training è completato
        if (total_errors == 0) {
            printf("Training completato dopo %d epoche!\n", epoch + 1);
            return epoch + 1;
        }
    }
    
    printf("Training terminato dopo %d epoche con %d errori\n", max_epochs, total_errors);
    return max_epochs;
}

// Stampa i pesi del percettrone
void perceptron_print_weights(Perceptron *p) {
    printf("Pesi: ");
    for (int i = 0; i < p->num_inputs; i++) {
        printf("w[%d]=%.3f ", i, p->weights[i]);
    }
    printf("bias=%.3f\n", p->bias);
}
