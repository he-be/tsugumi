// range_read: cold pread of byte ranges, in groups that run at the same time (docs/qfn-prefill/07 E2).
//   range_read <list file> <threads per group> <block MB>
// list lines: "<group> <file> <offset> <bytes>". Each group has its own thread pool (so the internal SSD and the
// external mirror are read at once, as the runtime would with two fds). Before reading, mincore() counts how much
// of each group is already resident (a warm page makes a fast read). Prints one line per group and the wall time.
// F_NOCACHE is not set: the runtime's preadRanges warms the page cache on purpose (docs/qwen38/05).
// cc -O2 -o range_read bench/qfn/range_read.c
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mach/mach_time.h>

static double now(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9; }
typedef struct { int fd; long off, len; } Blk;
typedef struct { int g; Blk* b; long nb, next; long blk; pthread_mutex_t mu; double done; long bytes, resident; } Group;
static Group G[8]; static int ng;
static char files[16][1024]; static int fds[16]; static char* maps[16]; static long sizes[16]; static int nf;
static double t0;

static int fileIndex(const char* path) {
  for (int i = 0; i < nf; i++) if (!strcmp(files[i], path)) return i;
  strcpy(files[nf], path); fds[nf] = open(path, O_RDONLY); if (fds[nf] < 0) { perror(path); exit(1); }
  struct stat sb; fstat(fds[nf], &sb); sizes[nf] = sb.st_size;
  maps[nf] = mmap(0, sb.st_size, PROT_READ, MAP_SHARED, fds[nf], 0);
  return nf++;
}
static long residentBytes(int f, long off, long len) {
  long ps = 16384, st = off / ps * ps, en = (off + len + ps - 1) / ps * ps, np = (en - st) / ps, tot = 0;
  char* v = malloc(np); mincore(maps[f] + st, en - st, v);
  for (long j = 0; j < np; j++) if (v[j] & 1) tot += ps;
  free(v); return tot;
}
static void* worker(void* p) {
  Group* g = p; char* buf = malloc(g->blk);
  for (;;) {
    pthread_mutex_lock(&g->mu); long i = g->next++; pthread_mutex_unlock(&g->mu);
    if (i >= g->nb) break;
    long o = g->b[i].off, l = g->b[i].len;
    while (l > 0) { ssize_t r = pread(g->b[i].fd, buf, l, o); if (r <= 0) { perror("pread"); exit(1); } o += r; l -= r; }
  }
  free(buf);
  pthread_mutex_lock(&g->mu); double t = now() - t0; if (t > g->done) g->done = t; pthread_mutex_unlock(&g->mu);
  return 0;
}
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "range_read <list> <threads per group> <block MB>\n"); return 2; }
  int th = atoi(argv[2]); long blk = atol(argv[3]) << 20;
  FILE* f = fopen(argv[1], "r"); if (!f) { perror(argv[1]); return 1; }
  int gid; char path[1024]; long off, len;
  while (fscanf(f, "%d %1023s %ld %ld", &gid, path, &off, &len) == 4) {
    int fi = fileIndex(path);
    Group* g = &G[gid]; if (gid + 1 > ng) ng = gid + 1;
    g->bytes += len; g->resident += residentBytes(fi, off, len);
    for (long p = 0; p < len; p += blk) {
      g->b = realloc(g->b, sizeof(Blk) * (g->nb + 1));
      g->b[g->nb++] = (Blk){ fds[fi], off + p, len - p < blk ? len - p : blk };
    }
  }
  pthread_t ts[8][64];
  for (int i = 0; i < ng; i++) { G[i].blk = blk; pthread_mutex_init(&G[i].mu, 0); }
  t0 = now();
  for (int i = 0; i < ng; i++) for (int t = 0; t < th; t++) pthread_create(&ts[i][t], 0, worker, &G[i]);
  for (int i = 0; i < ng; i++) for (int t = 0; t < th; t++) pthread_join(ts[i][t], 0);
  double wall = now() - t0; long bytes = 0, res = 0;
  for (int i = 0; i < ng; i++) {
    printf("group %d: %.3f GB (resident before %.3f GB) in %.3f s, %.2f GB/s\n", i, G[i].bytes / 1e9, G[i].resident / 1e9,
           G[i].done, G[i].bytes / 1e9 / G[i].done);
    bytes += G[i].bytes; res += G[i].resident;
  }
  printf("wall %.3f s for %.3f GB (resident before %.3f GB), %.2f GB/s, threads %d/group, block %ld MB\n",
         wall, bytes / 1e9, res / 1e9, bytes / 1e9 / wall, th, blk >> 20);
  return 0;
}
