/* montecarlo.c -- versión paralela con OpenMP
Uso: ./montecarlo <numero de muestras> <numero de hilos> */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    double start_time, end_time, exec_time;
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
    int p = (argc > 2) ? atoi(argv[2]) : 1;

    long dentro = 0;

    start_time = omp_get_wtime();

    omp_set_num_threads(p);
    
    #pragma omp parallel shared(p, N, dentro)
    {
        int id = omp_get_thread_num();
        
        /* Cada hilo recibe su semilla única espaciada al menos 2 * (N/p) valores,
           tal como pide el enunciado del taller. */
        unsigned int estado = 2230039u + (unsigned int)id * 2u * (N / p);

        #pragma omp for reduction(+:dentro) schedule(static)
        for (long long i = 0; i < N; i++) {
            double x = rand_r(&estado) / (double)RAND_MAX;
            double y = rand_r(&estado) / (double)RAND_MAX;
            if (x*x + y*y <= 1.0)
                dentro++;
        }
    }

    end_time = omp_get_wtime();
    exec_time = end_time - start_time;
    
    printf("Estimacion de pi: %.6f\n", 4.0 * dentro / (double)N);
    printf("Execution Time: %f seconds\n", exec_time);
    return 0;
}
