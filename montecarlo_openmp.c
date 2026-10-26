/* montecarlo.c-- version secuencial de referencia.
Uso: ./montecarlo <numero de muestras> */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    double start_time, end_time, exec_time;
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
    int p = (argc > 2) ? atol(argv[2]) : 1;

    long dentro = 0;

    start_time = omp_get_wtime();

    unsigned int estado = 2230039u;
    omp_set_num_threads(p);
    
    #pragma omp parallel default(none) firstprivate(estado) shared(p, N, dentro)
    {
      int id = omp_get_thread_num();
      estado += id * (N/p);
      printf("Hilo %d, Semilla %d\n", id, estado);

      long local = 0.0;

      #pragma omp for
      for (long i = 0; i < N; i++) {
          double x = rand_r(&estado) / (double)RAND_MAX;
          double y = rand_r(&estado) / (double)RAND_MAX;
          if (x*x + y*y <= 1.0)
              local++;
      }

      #pragma omp atomic
      dentro += local;
    }

    end_time = omp_get_wtime();
    exec_time = end_time - start_time;
    printf("Estimacion de pi: %.6f\n", 4.0 * dentro / (double)N);
    printf("Execution Time: %f seconds\n", exec_time);
    return 0;
}
