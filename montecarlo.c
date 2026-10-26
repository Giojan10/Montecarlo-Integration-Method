/* montecarlo.c-- version secuencial de referencia.
Uso: ./montecarlo <numero de muestras> */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    double start_time, end_time, exec_time;
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;

    unsigned int estado = 2230039u; /* semilla del generador */
    long dentro = 0;

    start_time = omp_get_wtime();
    for (long i = 0; i < N; i++) {
        double x = rand_r(&estado) / (double)RAND_MAX;
        double y = rand_r(&estado) / (double)RAND_MAX;
        if (x*x + y*y <= 1.0)
            dentro++;
    }

    end_time = omp_get_wtime();
    exec_time = end_time - start_time;
    printf("Estimacion de pi: %.6f\n", 4.0 * dentro / (double)N);
    printf("Execution Time: %f seconds\n", exec_time);
    return 0;
}
