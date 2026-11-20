// STREAM-style triad a[i] = b[i] + s*c[i] across T threads.
// Used as a synthetic "bandwidth hog" tool and as a bandwidth probe.
//
//   stream <threads> <MiB per array> <seconds>
// Prints achieved GB/s (3 arrays * 8 B per element per iteration).
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
  double *a, *b, *c;
  size_t n;
  double secs;
  double bytes;
} job_t;

static void *run(void *p) {
  job_t *j = p;
  for (size_t i = 0; i < j->n; i++) { j->b[i] = 1.0; j->c[i] = 2.0; j->a[i] = 0.0; }
  double t0 = now();
  while (now() - t0 < j->secs) {
    for (size_t i = 0; i < j->n; i++) j->a[i] = j->b[i] + 3.0 * j->c[i];
    j->bytes += 3.0 * 8.0 * j->n;
  }
  return NULL;
}

int main(int argc, char **argv) {
  int T = argc > 1 ? atoi(argv[1]) : 4;
  size_t mib = argc > 2 ? atol(argv[2]) : 256;
  double secs = argc > 3 ? atof(argv[3]) : 5.0;
  size_t n = mib * 1024 * 1024 / 8 / T;  // elements per array per thread

  pthread_t th[256];
  job_t jobs[256];
  double t0 = now();
  for (int t = 0; t < T; t++) {
    jobs[t] = (job_t){malloc(n * 8), malloc(n * 8), malloc(n * 8), n, secs, 0};
    pthread_create(&th[t], NULL, run, &jobs[t]);
  }
  double bytes = 0;
  for (int t = 0; t < T; t++) { pthread_join(th[t], NULL); bytes += jobs[t].bytes; }
  double el = now() - t0;
  printf("%.2f GB/s  (%d threads, %zu MiB/array, %.1f s)\n", bytes / el / 1e9, T, mib, el);
  return 0;
}
