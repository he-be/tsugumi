// Record-size x queue-depth random-read probe
// (QWEN38_FLASH_NEXT_VERIFY_PLAN.md D3: 1.49 / 1.84 / 2.76 MB at QD 1 / 10 / 20).
//
//   clang -O2 -o /tmp/rec_probe rec_probe.c -lpthread
//   /tmp/rec_probe <dir with layer_*.bin> <record bytes> <qd> <seconds>
//
// QD threads each issue one pread at a time at a random 16K-aligned offset in a
// random file, so the steady-state device queue depth is QD. NOCACHE=1 (default)
// bypasses the unified buffer cache; NOCACHE=0 RDAHEAD=1 reproduces production's
// fd flags. Prints GB/s plus per-request p50 / p99 / p999 latency.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>

#define MAX_FILES 64
#define MAX_THREADS 64
#define MAX_SAMPLES 400000

static int fds[MAX_FILES];
static off_t fsizes[MAX_FILES];
static int n_files = 0;
static size_t rec = 0;
static volatile int stop_flag = 0;

typedef struct {
    unsigned seed;
    char *buf;
    long count;
    double *lat;
    long n_lat;
} ctx_t;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void *worker(void *arg) {
    ctx_t *c = arg;
    while (!stop_flag) {
        int f = rand_r(&c->seed) % n_files;
        off_t span = fsizes[f] - (off_t)rec;
        off_t off = ((off_t)rand_r(&c->seed) * 16384 + (off_t)rand_r(&c->seed)) % span;
        off &= ~((off_t)16383);
        double t0 = now_ms();
        ssize_t r = pread(fds[f], c->buf, rec, off);
        double dt = now_ms() - t0;
        if (r != (ssize_t)rec) { perror("pread"); exit(1); }
        c->count++;
        if (c->n_lat < MAX_SAMPLES) c->lat[c->n_lat++] = dt;
    }
    return NULL;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "data";
    rec = argc > 2 ? (size_t)atol(argv[2]) : 2764800;
    int qd = argc > 3 ? atoi(argv[3]) : 1;
    double secs = argc > 4 ? atof(argv[4]) : 30.0;
    const char *e_nc = getenv("NOCACHE"), *e_ra = getenv("RDAHEAD");
    int nocache = e_nc ? atoi(e_nc) : 1;
    int rdahead = e_ra ? atoi(e_ra) : 0;
    if (qd > MAX_THREADS) return 1;

    char path[1024];
    for (int L = 0; L < MAX_FILES; ++L) {
        snprintf(path, sizeof path, "%s/layer_%02d.bin", dir, L);
        int fd = open(path, O_RDONLY);
        if (fd < 0) break;
        fcntl(fd, F_NOCACHE, nocache);
        fcntl(fd, F_RDAHEAD, rdahead);
        struct stat st; fstat(fd, &st);
        fsizes[n_files] = st.st_size;
        fds[n_files++] = fd;
    }
    if (n_files == 0) { fprintf(stderr, "no layer_*.bin under %s\n", dir); return 1; }

    pthread_t th[MAX_THREADS];
    ctx_t ctx[MAX_THREADS];
    for (int t = 0; t < qd; ++t) {
        ctx[t].seed = 12345u + 7919u * (unsigned)t;
        ctx[t].count = 0;
        ctx[t].n_lat = 0;
        ctx[t].lat = malloc(sizeof(double) * MAX_SAMPLES);
        if (posix_memalign((void **)&ctx[t].buf, 16384, rec)) return 1;
    }
    double t0 = now_ms();
    for (long t = 0; t < qd; ++t) pthread_create(&th[t], NULL, worker, &ctx[t]);
    struct timespec sl = { (time_t)secs, (long)((secs - (long)secs) * 1e9) };
    nanosleep(&sl, NULL);
    stop_flag = 1;
    for (int t = 0; t < qd; ++t) pthread_join(th[t], NULL);
    double elapsed = (now_ms() - t0) / 1e3;

    long total = 0, nl = 0;
    static double all[MAX_SAMPLES];
    for (int t = 0; t < qd; ++t) {
        total += ctx[t].count;
        for (long i = 0; i < ctx[t].n_lat && nl < MAX_SAMPLES; ++i) all[nl++] = ctx[t].lat[i];
    }
    qsort(all, nl, sizeof(double), cmp_d);
    double gbs = (double)total * rec / 1e9 / elapsed;
    double sum = 0;
    for (long i = 0; i < nl; ++i) sum += all[i];
    printf("%.2f\t%d\t%.3f\t%ld\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           rec / 1e6, qd, gbs, total, total / elapsed,
           all[(long)(nl * 0.10)], all[(long)(nl * 0.50)], sum / nl,
           all[(long)(nl * 0.99)], all[(long)(nl * 0.999)]);
    fflush(stdout);
    return 0;
}
