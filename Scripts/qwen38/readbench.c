// readbench <file> <ranges file> <first layer> <layers> <mode> <threads> <block MB>
// Cold read speed of expert ranges (ranges file from expert_ranges.py). docs/qwen38/05 §1.
// clang -O2 -o scratch/qwen38/readbench Scripts/qwen38/readbench.c
// mode: pread | touch (mmap, 1 byte per 16K page) | advise (F_RDADVISE then touch, 1 thread)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mach/mach_time.h>
typedef struct { long off, len; } Range;
static Range rs[1024]; static int nr;
static long blk; static int fd; static char* base; static const char* mode;
static long next_i = 0; static long total_blocks; static long blocks_off[1<<20], blocks_len[1<<20];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static double now(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
static void* worker(void* _) {
  char* buf = malloc(blk);
  for(;;){ pthread_mutex_lock(&mu); long i = next_i++; pthread_mutex_unlock(&mu); if(i>=total_blocks) break;
    if(!strcmp(mode,"pread")) { long o=blocks_off[i], l=blocks_len[i]; while(l>0){ ssize_t r=pread(fd,buf,l,o); if(r<=0)break; o+=r; l-=r; } }
    else { volatile char s=0; for(long p=blocks_off[i]; p<blocks_off[i]+blocks_len[i]; p+=16384) s+=base[p]; }
  }
  free(buf); return 0;
}
static long resident(void){ long tot=0; for(int i=0;i<nr;i++){ long ps=16384, st=rs[i].off/ps*ps, en=(rs[i].off+rs[i].len+ps-1)/ps*ps; long np=(en-st)/ps; char* v=malloc(np); mincore(base+st,en-st,v); for(long j=0;j<np;j++) if(v[j]&1) tot+=ps; free(v);} return tot; }
int main(int argc,char**argv){
  fd=open(argv[1],O_RDONLY); struct stat sb; fstat(fd,&sb);
  base=mmap(0,sb.st_size,PROT_READ,MAP_SHARED,fd,0);
  FILE* f=fopen(argv[2],"r"); int first=atoi(argv[3]), layers=atoi(argv[4]); mode=argv[5]; int th=atoi(argv[6]); blk=atol(argv[7])<<20;
  long o,n; char name[256];
  while(fscanf(f,"%ld %ld %255s",&o,&n,name)==3){ int L; if(sscanf(name,"blk.%d.",&L)==1 && L>=first && L<first+layers){ rs[nr].off=o; rs[nr].len=n; nr++; } }
  long bytes=0; for(int i=0;i<nr;i++){ bytes+=rs[i].len; for(long p=0;p<rs[i].len;p+=blk){ blocks_off[total_blocks]=rs[i].off+p; blocks_len[total_blocks]=(rs[i].len-p<blk)?rs[i].len-p:blk; total_blocks++; } }
  double r0=resident()/1e9;
  double t0=now(), tadv=0;
  if(!strcmp(mode,"advise")){ for(int i=0;i<nr;i++){ struct radvisory ra={rs[i].off,(int)rs[i].len}; fcntl(fd,F_RDADVISE,&ra);} tadv=now()-t0; }
  pthread_t ts[64]; for(int i=0;i<th;i++) pthread_create(&ts[i],0,worker,0); for(int i=0;i<th;i++) pthread_join(ts[i],0);
  double dt=now()-t0;
  printf("%s layers %d..%d %.2f GB resident before %.2f GB, threads %d block %ld MB: %.2f s (advise %.2f s), %.2f GB/s of cold\n",
    mode, first, first+layers-1, bytes/1e9, r0, th, blk>>20, dt, tadv, (bytes/1e9-r0)/dt);
}
