// Synthesise a packed_experts-shaped dataset: <dir> <layers> <bytes per layer>.
// Content is xorshift noise (never sparse, never a clone) so random reads hit NAND.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "data";
    int layers = argc > 2 ? atoi(argv[2]) : 31;
    long bytes = argc > 3 ? atol(argv[3]) : 429916160L;
    size_t chunk = 8u << 20;
    uint64_t *buf = malloc(chunk);
    if (!buf) return 1;
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    char path[1024];
    for (int L = 0; L < layers; ++L) {
        snprintf(path, sizeof path, "%s/layer_%02d.bin", dir, L);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { perror(path); return 1; }
        fcntl(fd, F_NOCACHE, 1);
        long left = bytes;
        while (left > 0) {
            size_t n = (size_t)(left < (long)chunk ? left : (long)chunk);
            for (size_t i = 0; i < n / 8; ++i) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                buf[i] = s;
            }
            ssize_t w = write(fd, buf, n);
            if (w != (ssize_t)n) { perror("write"); return 1; }
            left -= n;
        }
        fsync(fd);
        close(fd);
        fprintf(stderr, "%s %ld bytes\n", path, bytes);
    }
    return 0;
}
