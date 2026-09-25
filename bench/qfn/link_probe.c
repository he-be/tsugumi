// link_probe: raw TCP over the TB4 bridge (docs/qfn-prefill/07 E0).
//   server:  link_probe serve <bind ip> <port>
//   client:  link_probe run <server ip> <port> <reps>
//   client:  link_probe stream <server ip> <port> <seconds>   1 GiB acked sends back to back for <seconds> (19 §2-3-1)
// The client measures, each rep in turn on one connection:
//   l_1   84 MB one way (client -> server, server acks 1 byte)   = A's chunk (2048 x 4 x 2560 fp32)
//   l_48  48 round trips of 21 MB out + 21 MB back              = B's chunk (2048 x 2560 fp32 each way)
//   bulk  1 GiB one way (acked)
//   rtt   1-byte ping-pong, median of 200
// cc -O2 -o bench/qfn/link_probe bench/qfn/link_probe.c
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <mach/mach_time.h>

static double now(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9; }
static void sendall(int fd, const char* p, long n) { while (n > 0) { ssize_t r = send(fd, p, n > (1 << 24) ? (1 << 24) : n, 0);
  if (r <= 0) { perror("send"); exit(1); } p += r; n -= r; } }
static void recvall(int fd, char* p, long n) { while (n > 0) { ssize_t r = recv(fd, p, n > (1 << 24) ? (1 << 24) : n, 0);
  if (r <= 0) { perror("recv"); exit(1); } p += r; n -= r; } }
static void tune(int fd) { int one = 1, buf = 8 << 20; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf); setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf); }
enum { L1 = 2048L * 4 * 2560 * 4, L48 = 2048L * 2560 * 4, BULK = 1L << 30 };
// Protocol: client sends an 8-byte header {op, bytes}; op 1 = sink `bytes` then ack 1 byte; op 2 = sink `bytes` then
// send `bytes` back; op 3 = 1-byte echo; op 0 = done.
typedef struct { long op, bytes; } Hdr;

static int cmp(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: see header\n"); return 2; }
  char* buf = malloc(BULK);
  memset(buf, 7, BULK);
  struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(atoi(argv[3])) };
  inet_pton(AF_INET, argv[2], &a.sin_addr);
  if (!strcmp(argv[1], "serve")) {
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (struct sockaddr*)&a, sizeof a) || listen(ls, 1)) { perror("bind/listen"); return 1; }
    printf("listening on %s:%s\n", argv[2], argv[3]); fflush(stdout);
    int fd = accept(ls, 0, 0); tune(fd);
    for (;;) { Hdr h; recvall(fd, (char*)&h, sizeof h);
      if (h.op == 0) break;
      if (h.op == 1) { recvall(fd, buf, h.bytes); sendall(fd, buf, 1); }
      else if (h.op == 2) { recvall(fd, buf, h.bytes); sendall(fd, buf, h.bytes); }
      else if (h.op == 3) { recvall(fd, buf, 1); sendall(fd, buf, 1); } }
    close(fd); close(ls); return 0;
  }
  int reps = argc > 4 ? atoi(argv[4]) : 3;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (connect(fd, (struct sockaddr*)&a, sizeof a)) { perror("connect"); return 1; }
  tune(fd);
  if (!strcmp(argv[1], "stream")) {
    double secs = argc > 4 ? atof(argv[4]) : 15, t0 = now(), bytes = 0; int n = 0;
    printf("gib\ts\tGBps\n");
    while (now() - t0 < secs) { Hdr h = { 1, BULK }; double t = now(); sendall(fd, (char*)&h, sizeof h); sendall(fd, buf, BULK); recvall(fd, buf, 1);
      double d = now() - t; bytes += BULK; printf("%d\t%.4f\t%.2f\n", ++n, d, BULK / d / 1e9); fflush(stdout); }
    printf("total\t%.3f\t%.2f\n", now() - t0, bytes / (now() - t0) / 1e9);
    Hdr h = { 0, 0 }; sendall(fd, (char*)&h, sizeof h); close(fd); return 0;
  }
  double rtts[200];
  for (int i = 0; i < 200; i++) { Hdr h = { 3, 1 }; double t = now(); sendall(fd, (char*)&h, sizeof h); sendall(fd, buf, 1); recvall(fd, buf, 1); rtts[i] = now() - t; }
  qsort(rtts, 200, sizeof(double), cmp);
  printf("rtt_ms median %.3f p10 %.3f p90 %.3f\n", rtts[100] * 1e3, rtts[20] * 1e3, rtts[180] * 1e3);
  printf("rep\tl1_s\tl48_s\tbulk_s\tbulk_GBps\n");
  for (int r = 0; r < reps; r++) {
    Hdr h = { 1, L1 }; double t = now(); sendall(fd, (char*)&h, sizeof h); sendall(fd, buf, L1); recvall(fd, buf, 1); double l1 = now() - t;
    t = now();
    for (int i = 0; i < 48; i++) { Hdr h2 = { 2, L48 }; sendall(fd, (char*)&h2, sizeof h2); sendall(fd, buf, L48); recvall(fd, buf, L48); }
    double l48 = now() - t;
    Hdr h3 = { 1, BULK }; t = now(); sendall(fd, (char*)&h3, sizeof h3); sendall(fd, buf, BULK); recvall(fd, buf, 1); double bulk = now() - t;
    printf("%d\t%.4f\t%.4f\t%.4f\t%.2f\n", r + 1, l1, l48, bulk, BULK / bulk / 1e9);
    fflush(stdout);
  }
  Hdr h = { 0, 0 }; sendall(fd, (char*)&h, sizeof h);
  close(fd); return 0;
}
