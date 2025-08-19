#include <stdio.h>
#include <cuda_runtime.h>

// Kernel CUDA che viene eseguito sulla GPU
__global__ void helloKernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello World dal thread %d!\n", idx);
}

int main() {
    printf("Hello World dalla CPU (host)!\n");
    
    // Configurazione per il lancio del kernel
    int numThreads = 8;
    int numBlocks = 2;
    
    // Lancio del kernel sulla GPU
    helloKernel<<<numBlocks, numThreads>>>();
    
    // Sincronizzazione: aspetta che tutti i thread GPU finiscano
    cudaDeviceSynchronize();
    
    // Controlla eventuali errori CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Errore CUDA: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("Programma completato!\n");
    return 0;
}