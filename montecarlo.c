/* montecarlo_omp.c -- Monte Carlo para pi con rand_r, reproducible para cualquier p.
   Uso: ./montecarlo_omp <N> <p>
   Nota: depende de la implementacion de rand_r de glibc (3 pasos LCG por llamada). */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define SEMILLA_BASE 2230039u
#define LCG_A 1103515245u
#define LCG_C 12345u

/* Avanza el estado n pasos del LCG en O(log n) */
static unsigned int lcg_salto(unsigned int s, uint64_t n) {
    unsigned int acc_m = 1, acc_p = 0, cur_m = LCG_A, cur_p = LCG_C;
    while (n) {
        if (n & 1) {
            acc_m *= cur_m;
            acc_p = acc_p * cur_m + cur_p;
        }
        cur_p = (cur_m + 1) * cur_p;
        cur_m *= cur_m;
        n >>= 1;
    }
    return acc_m * s + acc_p;
}

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
    int  p = (argc > 2) ? atoi(argv[2]) : 1;

    long dentro = 0;

    omp_set_num_threads(p);
    double t0 = omp_get_wtime();

    unsigned int estado = SEMILLA_BASE;
    long siguiente = -1;

    #pragma omp parallel for schedule(static) reduction(+:dentro) \
                             firstprivate(estado, siguiente)
    for (long i = 0; i < N; i++) {
        if (i != siguiente) {   /* primer iteracion del bloque del hilo */
            estado = lcg_salto(SEMILLA_BASE, 6ULL * (uint64_t)i);
            printf("Hilo %d, iteracion inicial %ld, semilla %u\n",
                   omp_get_thread_num(), i, estado);
        }

        double x = rand_r(&estado) / (double)RAND_MAX;
        double y = rand_r(&estado) / (double)RAND_MAX;
        siguiente = i + 1;

        if (x*x + y*y <= 1.0)
            dentro++;
    }

    double t = omp_get_wtime() - t0;
    printf("Hilos: %d, N: %ld, dentro: %ld\n", p, N, dentro);
    printf("Estimacion de pi: %.6f\n", 4.0 * dentro / (double)N);
    printf("Execution Time: %f seconds\n", t);
    return 0;
}
