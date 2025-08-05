#include "perceptron.h"

void test_perceptron(Dataset *data, const char *name) {
    printf("\n=== Test %s ===\n", name);
    
    // Crea il percettrone
    Perceptron *p = perceptron_create(2, 0.1);
    if (!p) {
        printf("Errore nella creazione del percettrone\n");
        return;
    }
    
    printf("Pesi iniziali: ");
    perceptron_print_weights(p);
    
    // Training
    int epochs = perceptron_train_dataset(p, data, 100);
    
    printf("Pesi finali: ");
    perceptron_print_weights(p);
    
    // Test delle predizioni
    printf("\nTest delle predizioni:\n");
    for (int i = 0; i < data->num_samples; i++) {
        int prediction = perceptron_predict(p, data->inputs[i]);
        printf("Input: [%.0f, %.0f] -> Predizione: %d, Target: %d %s\n",
               data->inputs[i][0], data->inputs[i][1],
               prediction, data->targets[i],
               prediction == data->targets[i] ? "✓" : "✗");
    }
    
    perceptron_destroy(p);
}

int main() {
    printf("=== Libreria Percettrone in C ===\n");
    
    // Test funzione AND
    Dataset *and_data = dataset_create_and();
    if (and_data) {
        test_perceptron(and_data, "AND");
        dataset_destroy(and_data);
    }
    
    // Test funzione OR
    Dataset *or_data = dataset_create_or();
    if (or_data) {
        test_perceptron(or_data, "OR");
        dataset_destroy(or_data);
    }
    
    // Test funzione XOR (dovrebbe fallire!)
    Dataset *xor_data = dataset_create_xor();
    if (xor_data) {
        test_perceptron(xor_data, "XOR (dovrebbe fallire)");
        dataset_destroy(xor_data);
    }
    
    return 0;
}
