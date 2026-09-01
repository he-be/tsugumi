// Expert-fetch queue-depth probe (docs/mtp/33-M8-IO-FLOOR.md §1).
//
// 27-M7 §7 swept pread bandwidth against "how many experts are read at once"
// and concluded the SSD is the floor. That probe was not committed, it left
// the unified buffer cache on, and it swept the split (deeper queue at fixed
// bytes) only at 4 concurrent experts. Production issues exactly the layer's
// miss count at once -- `PreadExpertStreamer.executeExpertCachePlan` uses
// `DispatchQueue.concurrentPerform(iterations: plan.misses.count)` -- which is
// 1.5 to 4.7 reads depending on the slot count, so the shallow rows are the
// ones that matter and they were the ones missing.
//
//   clang -O2 -o /tmp/io_depth_probe bench/io_depth_probe.c -lpthread
//   /tmp/io_depth_probe scratch/gemma4-qat.moepack/packed_experts 30 128
//
// argv: <packed_experts dir> [trials] [experts per layer].
// NOCACHE=1 (default) sets F_NOCACHE so the numbers are the device, not the
// page cache; NOCACHE=0 RDAHEAD=1 reproduces production's fd flags. Reads are
// random (layer, expert) so a trial does not replay the previous one.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>

#define MAX_LAYERS 64
#define MAX_REQ 256

static int fds[MAX_LAYERS];
static int n_layers = 0;
static long expert_stride = 0;
static int experts_per_layer = 0;

typedef struct { int fd; off_t off; size_t len; char *dst; } req_t;

static req_t reqs[MAX_REQ];
static int n_reqs = 0;


// macOS has no pthread_barrier_t; a mutex + condvar barrier is enough here.
typedef struct {
    pthread_mutex_t m; pthread_cond_t c;
    unsigned count, target, generation;
} barrier_t;

static void barrier_init(barrier_t *b, unsigned target) {
    pthread_mutex_init(&b->m, NULL); pthread_cond_init(&b->c, NULL);
    b->count = 0; b->target = target; b->generation = 0;
}
static void barrier_destroy(barrier_t *b) {
    pthread_mutex_destroy(&b->m); pthread_cond_destroy(&b->c);
}
static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->m);
    unsigned gen = b->generation;
    if (++b->count == b->target) {
        b->count = 0; b->generation++;
        pthread_cond_broadcast(&b->c);
    } else {
        while (gen == b->generation) pthread_cond_wait(&b->c, &b->m);
    }
    pthread_mutex_unlock(&b->m);
}

static barrier_t bar_start, bar_end;

static int n_threads = 0;
static int stop_flag = 0;

static void *worker(void *arg) {
    long id = (long)arg;
    for (;;) {
        barrier_wait(&bar_start);
        if (stop_flag) return NULL;
        for (int i = (int)id; i < n_reqs; i += n_threads) {
            size_t filled = 0;
            while (filled < reqs[i].len) {
                ssize_t r = pread(reqs[i].fd, reqs[i].dst + filled,
                                  reqs[i].len - filled, reqs[i].off + filled);
                if (r <= 0) { perror("pread"); exit(1); }
                filled += (size_t)r;
            }
        }
        barrier_wait(&bar_end);
    }
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "scratch/gemma4-qat.moepack/packed_experts";
    int trials = argc > 2 ? atoi(argv[2]) : 40;
    const char *e_nc = getenv("NOCACHE"), *e_ra = getenv("RDAHEAD");
    int nocache = e_nc ? atoi(e_nc) : 1;
    int rdahead = e_ra ? atoi(e_ra) : 0;
    experts_per_layer = argc > 3 ? atoi(argv[3]) : 128;

    char path[1024];
    for (int L = 0; L < MAX_LAYERS; ++L) {
        snprintf(path, sizeof path, "%s/layer_%02d.bin", dir, L);
        int fd = open(path, O_RDONLY);
        if (fd < 0) break;
        fcntl(fd, F_NOCACHE, nocache);    // 1 = bypass the unified buffer cache
        fcntl(fd, F_RDAHEAD, rdahead);
        if (L == 0) {
            struct stat st; fstat(fd, &st);
            expert_stride = (long)st.st_size / experts_per_layer;
        }
        fds[n_layers++] = fd;
    }
    if (n_layers == 0) { fprintf(stderr, "no layer files under %s\n", dir); return 1; }
    fprintf(stderr, "layers=%d expert_stride=%ld (%.2f MB) nocache=%d rdahead=%d\n",
            n_layers, expert_stride, expert_stride / 1e6, nocache, rdahead);

    char *buf = NULL;
    if (posix_memalign((void **)&buf, 16384, (size_t)expert_stride * 16)) return 1;

    int concs[] = {1, 2, 3, 4, 6, 8, 12, 16};
    int splits[] = {1, 2, 4};
    int n_conc = sizeof concs / sizeof concs[0];
    int n_split = sizeof splits / sizeof splits[0];

    printf("conc\tsplit\tdepth\tMB\tmed_ms\tGB/s\n");
    srandom(12345);

    for (int ci = 0; ci < n_conc; ++ci) {
        for (int si = 0; si < n_split; ++si) {
            int E = concs[ci], S = splits[si];
            int depth = E * S;
            if (depth > MAX_REQ) continue;
            n_threads = depth;
            stop_flag = 0;
            barrier_init(&bar_start, n_threads + 1);
            barrier_init(&bar_end, n_threads + 1);
            pthread_t th[MAX_REQ];
            for (long t = 0; t < n_threads; ++t) pthread_create(&th[t], NULL, worker, (void *)t);

            double samples[512];
            int used = 0;
            for (int tr = 0; tr < trials; ++tr) {
                int L = (int)(random() % n_layers);
                n_reqs = 0;
                size_t chunk = (size_t)expert_stride / S;
                for (int e = 0; e < E; ++e) {
                    int ex = (int)(random() % experts_per_layer);
                    for (int s = 0; s < S; ++s) {
                        size_t len = (s == S - 1) ? (size_t)expert_stride - chunk * (S - 1) : chunk;
                        reqs[n_reqs].fd = fds[L];
                        reqs[n_reqs].off = (off_t)ex * expert_stride + (off_t)(chunk * s);
                        reqs[n_reqs].len = len;
                        reqs[n_reqs].dst = buf + (size_t)e * expert_stride + chunk * s;
                        n_reqs++;
                    }
                }
                double t0 = now_ms();
                barrier_wait(&bar_start);
                barrier_wait(&bar_end);
                double dt = now_ms() - t0;
                if (tr >= 4) samples[used++] = dt;   // drop warmup
            }
            stop_flag = 1;
            barrier_wait(&bar_start);
            for (int t = 0; t < n_threads; ++t) pthread_join(th[t], NULL);
            barrier_destroy(&bar_start);
            barrier_destroy(&bar_end);

            qsort(samples, used, sizeof(double), cmp_d);
            double med = samples[used / 2];
            double mb = (double)E * expert_stride / 1e6;
            printf("%d\t%d\t%d\t%.2f\t%.3f\t%.2f\n", E, S, depth, mb, med, mb / med);
            fflush(stdout);
        }
    }
    return 0;
}
