/* montecarlo_omp.c -- Monte Carlo para pi, reproducible para cualquier p.
   Uso: ./montecarlo_omp <N> <p> */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define SEMILLA_BASE 2230039ULL          /* semilla asignada por el docente */
#define LCG_A 6364136223846793005ULL     /* LCG de 64 bits (Knuth MMIX) */
#define LCG_C 1442695040888963407ULL

static inline uint64_t lcg_paso(uint64_t s) {
    return LCG_A * s + LCG_C;
}

/* Avanza el estado n posiciones en O(log n) (composicion de mapas afines) */
static uint64_t lcg_salto(uint64_t s, uint64_t n) {
    uint64_t acc_m = 1, acc_p = 0, cur_m = LCG_A, cur_p = LCG_C;
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

/* Usa los 53 bits altos (los bajos de un LCG son de mala calidad) */
static inline double a_unitario(uint64_t s) {
    return (s >> 11) * (1.0 / 9007199254740992.0);   /* 2^-53 */
}

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
    int  p = (argc > 2) ? atoi(argv[2]) : 1;

    long dentro = 0;

    omp_set_num_threads(p);
    double t0 = omp_get_wtime();

    /* estado: flujo privado del hilo; siguiente: proxima iteracion esperada.
       Si el hilo recibe un bloque nuevo (i != siguiente), salta a la
       posicion 2*i de la secuencia global. */
    uint64_t estado = SEMILLA_BASE;
    long siguiente = -1;

    #pragma omp parallel for schedule(static) reduction(+:dentro) \
                             firstprivate(estado, siguiente)
    for (long i = 0; i < N; i++) {
        if (i != siguiente)
            estado = lcg_salto(SEMILLA_BASE, 2ULL * (uint64_t)i);

        estado = lcg_paso(estado);  double x = a_unitario(estado);
        estado = lcg_paso(estado);  double y = a_unitario(estado);
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
