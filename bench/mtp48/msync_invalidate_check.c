#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>
static long incore(char *b, size_t n){
  size_t pages=n/16384; char *v=malloc(pages);
  mincore(b,n,v); long c=0; for(size_t i=0;i<pages;i++) if(v[i]&1) c++; free(v); return c;
}
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
int main(int argc,char**argv){
  int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st); size_t n=st.st_size;
  // pass 1: map, touch, invalidate, unmap
  char *b=mmap(0,n,PROT_READ,MAP_SHARED|MAP_FILE,fd,0);
  volatile char x=0; for(size_t i=0;i<n;i+=16384) x^=b[i];
  printf("pass1 incore after touch: %ld\n", incore(b,n));
  msync(b,n,MS_INVALIDATE); munmap(b,n);
  // pass 2: fresh mapping -- is the UBC really empty?
  char *c2=mmap(0,n,PROT_READ,MAP_SHARED|MAP_FILE,fd,0);
  printf("pass2 incore on fresh map: %ld\n", incore(c2,n));
  double t0=now(); for(size_t i=0;i<n;i+=16384) x^=c2[i]; double t1=now();
  printf("pass2 cold serial touch : %.3f s -> %.2f GB/s, incore now %ld\n",
         t1-t0, n/(t1-t0)/1e9, incore(c2,n));
  double t2=now(); for(size_t i=0;i<n;i+=16384) x^=c2[i]; double t3=now();
  printf("pass2 warm serial touch : %.3f s -> %.2f GB/s\n", t3-t2, n/(t3-t2)/1e9);
  (void)x; return 0;
}
