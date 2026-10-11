/* Native x86-64 cumulative diagnostic reader; provenance checks inherited from recording-reader.
 * Snapshots are not transactional: sample_start/end bound their collection span. */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/un.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(__x86_64__)
#error Supported only on Linux x86-64
#endif
#define FILE_BYTES (4096+256*512)
struct __attribute__((aligned(64))) row {
 char name[384]; uint64_t function,module,tls_context,stream,native_tid; char padding[24];
 _Atomic uint64_t published,success,failure,last_error;
};
_Static_assert(sizeof(struct row)==512 && offsetof(struct row,published)==448,"row ABI");
struct __attribute__((aligned(64))) header {
    char magic[8]; uint32_t schema, bytes, pid, reserved; uint64_t start_ticks;
    char boot[36], app[32], build[64], instance[32], padding[60];
    _Atomic uint64_t active, total, failed, dropped, unknown;
};
_Static_assert(offsetof(struct header, active)==256, "active ABI");
_Static_assert(offsetof(struct header, total)==264, "counter ABI");
_Static_assert(sizeof(struct header)==320 && ATOMIC_LLONG_LOCK_FREE==2, "atomic ABI");
static volatile sig_atomic_t stopping;
static void stop(int signal_number) { (void)signal_number; stopping=1; }
static double now(void) { struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) exit(2); return t.tv_sec+t.tv_nsec/1e9; }
static int text_file(const char *path,char *out,size_t size) {
    int fd=open(path,O_RDONLY|O_CLOEXEC); if(fd<0) return -1;
    ssize_t n=read(fd,out,size-1); close(fd); if(n<0 || (size_t)n==size-1) return -1;
    out[n]=0; return (int)n;
}
static int identity(int pid,uint64_t *ticks,uint32_t *namespace_pid) {
    char path[128],s[8192]; snprintf(path,sizeof path,"/proc/%d/stat",pid);
    if(text_file(path,s,sizeof s)<0) return -1;
    char *p=strrchr(s,')'); if(!p) return -1;
    char *save=NULL,*token=strtok_r(p+1," ",&save);
    for(int i=0;i<19 && token;i++) token=strtok_r(NULL," ",&save);
    if(!token) return -1;
    char *end; errno=0; *ticks=strtoull(token,&end,10);
    if(errno || (*end && *end!='\n')) return -1;
    snprintf(path,sizeof path,"/proc/%d/status",pid);
    if(text_file(path,s,sizeof s)<0) return -1;
    p=strstr(s,"\nNSpid:"); if(!p) return -1; p+=7;
    unsigned long last=0;
    while(*p && *p!='\n') { while(*p==' '||*p=='\t') ++p; if(*p=='\n'||!*p) break;
        errno=0; last=strtoul(p,&end,10); if(errno||end==p||last>UINT32_MAX) return -1; p=end; }
    if(!last) return -1;
    *namespace_pid=(uint32_t)last; return 0;
}
static int game_identity(int pid,const char *expected) {
    char path[128],s[8192]; snprintf(path,sizeof path,"/proc/%d/cmdline",pid);
    if(text_file(path,s,sizeof s)<0) return -1;
    char *base=s; for(char *p=s;*p;p++) if(*p=='/'||*p=='\\') base=p+1;
    return strcmp(base,expected) ? -1:0;
}
static int private_object(int fd,int directory) {
    struct stat s; if(fstat(fd,&s)||s.st_uid!=getuid()) return -1;
    return directory ? (!S_ISDIR(s.st_mode)||(s.st_mode&07777)!=0700 ? -1:0)
        : (!S_ISREG(s.st_mode)||(s.st_mode&07777)!=0600||s.st_size!=FILE_BYTES ? -1:0);
}
static int mapped_library(int pid,const struct stat *library) {
    char path[128],line[8192]; snprintf(path,sizeof path,"/proc/%d/maps",pid);
    FILE *f=fopen(path,"re"); if(!f) return -1; int found=0;
    while(fgets(line,sizeof line,f)) {
        unsigned long lo,hi,off; unsigned maj,min; unsigned long long ino; char perms[5];
        if(sscanf(line,"%lx-%lx %4s %lx %x:%x %llu",&lo,&hi,perms,&off,&maj,&min,&ino)==7
            && ino==(unsigned long long)library->st_ino && makedev(maj,min)==library->st_dev && perms[2]=='x') found=1;
    }
    fclose(f); return found?0:-1;
}
static int hash_library(int fd,const char *expected) {
    if(strlen(expected)!=64) return -1;
    for(int i=0;i<64;i++) if(!isxdigit((unsigned char)expected[i])) return -1;
    int pipefd[2]; if(pipe2(pipefd,O_CLOEXEC)) return -1;
    pid_t child=fork(); if(child<0) { close(pipefd[0]);close(pipefd[1]);return -1; }
    if(!child) {
        if(dup2(fd,STDIN_FILENO)<0||dup2(pipefd[1],STDOUT_FILENO)<0) _exit(126);
        execl("/usr/bin/sha256sum","sha256sum",(char *)NULL); _exit(127);
    }
    close(pipefd[1]); char digest[128]; ssize_t n=read(pipefd[0],digest,sizeof digest); close(pipefd[0]);
    int status=-1; while(waitpid(child,&status,0)<0) if(errno!=EINTR) return -1;
    return n>=64 && WIFEXITED(status)&&WEXITSTATUS(status)==0&&!strncasecmp(digest,expected,64) ? 0:-1;
}
static int hex_bytes(const char *s,size_t n) { for(size_t i=0;i<n;i++) if(!isxdigit((unsigned char)s[i])) return 0; return 1; }
struct target {
    int pid,pidfd,directory,library_fd; uint32_t namespace_pid; uint64_t ticks;
    const char *app,*build,*game,*library_path; char boot[36],instance[32];
    struct stat library,record; struct header *header;
};
static int same_file(const struct stat *a,const struct stat *b) {
    return a->st_dev==b->st_dev&&a->st_ino==b->st_ino&&a->st_size==b->st_size
        &&a->st_mtim.tv_sec==b->st_mtim.tv_sec&&a->st_mtim.tv_nsec==b->st_mtim.tv_nsec
        &&a->st_ctim.tv_sec==b->st_ctim.tv_sec&&a->st_ctim.tv_nsec==b->st_ctim.tv_nsec;
}
static int library_unchanged(const struct target *t) {
    struct stat s,path;
    return !fstat(t->library_fd,&s)&&same_file(&s,&t->library)
        &&!lstat(t->library_path,&path)&&same_file(&path,&t->library)&&!mapped_library(t->pid,&t->library);
}
static int live(const struct target *t) {
    struct pollfd p={t->pidfd,POLLIN,0}; uint64_t ticks; uint32_t ns;
    if(poll(&p,1,0)!=0 || identity(t->pid,&ticks,&ns) || ticks!=t->ticks || ns!=t->namespace_pid) return 0;
    return !game_identity(t->pid,t->game);
}
static int instance_matches(const struct target *t,const struct header *h) {
    if(atomic_load_explicit(&h->active,memory_order_acquire)!=1) return 0;
    if(memcmp(h->magic,"ZLUDAL01",8)||h->schema!=1||h->bytes!=FILE_BYTES||h->reserved) return -1;
    if(h->pid!=t->namespace_pid||h->start_ticks!=t->ticks) return 0;
    if(memcmp(h->boot,t->boot,36)||!memchr(h->app,0,sizeof h->app)||strcmp(h->app,t->app)
        ||!memchr(h->build,0,sizeof h->build)||strcmp(h->build,t->build)||!hex_bytes(h->instance,32)) return -1;
    return 1;
}
/* Re-enumerate so a second active library instance is never silently aggregated. */
static int select_instance(struct target *t) {
    int copied=openat(t->directory,".",O_RDONLY|O_DIRECTORY|O_CLOEXEC); if(copied<0) return -1;
    DIR *d=fdopendir(copied); if(!d) { close(copied); return -1; }
    int count=0,bad=0; struct header *selected=NULL; struct stat selected_file={0}; struct dirent *e;
    while((e=readdir(d))) {
        size_t n=strlen(e->d_name); if(n<7||strcmp(e->d_name+n-7,".survey")) continue;
        int fd=openat(t->directory,e->d_name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);
        if(fd<0) { if(errno==ENOENT) continue; bad=1;break; }
        if(private_object(fd,0)) {close(fd);bad=1;break;}
        struct stat opened;if(fstat(fd,&opened)){close(fd);bad=1;break;}
        struct header *h=mmap(NULL,FILE_BYTES,PROT_READ,MAP_SHARED,fd,0); close(fd);
        if(h==MAP_FAILED) {bad=1;break;}
        int match=instance_matches(t,h);
        if(match<0) bad=1;
        if(match==1) { count++; if(!selected) {selected=h;selected_file=opened;} else munmap(h,FILE_BYTES); }
        else munmap(h,FILE_BYTES);
        if(bad) break;
    }
    closedir(d);
    if(bad||count!=1) {if(selected)munmap(selected,FILE_BYTES);return -1;}
    if(t->header) {
        int same=!memcmp(selected->instance,t->instance,32)
            &&selected_file.st_dev==t->record.st_dev&&selected_file.st_ino==t->record.st_ino;
        munmap(selected,FILE_BYTES); return same?0:-1;
    }
    t->header=selected;t->record=selected_file;memcpy(t->instance,selected->instance,32); return 0;
}
static void json_string(const char *value) {
 putchar('"');
 for(const unsigned char *p=(const unsigned char *)value;*p;p++) {
  if(*p=='"'||*p=='\\') {putchar('\\');putchar(*p);}
  else if(*p<32) printf("\\u%04x",*p);
  else putchar(*p);
 }
 putchar('"');
}
/* Return 1 for normal unload/death, -1 for malformed rows or output failure. */
static int snapshot(const struct target *t) {
 struct saved_row { char name[384]; uint64_t function,module,context,stream,tid,success,failure,error; unsigned slot; } saved[256];
 unsigned count=0;
 double begin=now(); uint64_t total=atomic_load_explicit(&t->header->total,memory_order_relaxed);
 uint64_t failed=atomic_load_explicit(&t->header->failed,memory_order_relaxed);
 uint64_t dropped=atomic_load_explicit(&t->header->dropped,memory_order_relaxed);
 uint64_t unknown=atomic_load_explicit(&t->header->unknown,memory_order_relaxed);
 struct row *rows=(void *)((char *)t->header+4096);
 for(unsigned i=0;i<256;i++) {
  uint64_t published=atomic_load_explicit(&rows[i].published,memory_order_acquire);
  if(!published)continue;
  if(published!=1||!memchr(rows[i].name,0,sizeof rows[i].name))return -1;
  struct saved_row *r=&saved[count++];
  memcpy(r->name,rows[i].name,sizeof r->name);r->slot=i;
  r->function=rows[i].function;r->module=rows[i].module;r->context=rows[i].tls_context;
  r->stream=rows[i].stream;r->tid=rows[i].native_tid;
  r->success=atomic_load_explicit(&rows[i].success,memory_order_relaxed);
  r->failure=atomic_load_explicit(&rows[i].failure,memory_order_relaxed);
  r->error=atomic_load_explicit(&rows[i].last_error,memory_order_relaxed);
 }
 double finish=now();
 // Do not publish a partial snapshot or old values after physical unload/death.
 if(!live(t)||atomic_load_explicit(&t->header->active,memory_order_acquire)!=1)return 1;
 printf("{\"pid\":%d,\"start_ticks\":%llu,\"sample_start\":%.9f,\"sample_end\":%.9f,\"total_success\":%llu,\"total_failure\":%llu,\"dropped\":%llu,\"unknown\":%llu,\"rows\":[",t->pid,(unsigned long long)t->ticks,begin,finish,(unsigned long long)total,(unsigned long long)failed,(unsigned long long)dropped,(unsigned long long)unknown);
 for(unsigned i=0;i<count;i++) {
  const struct saved_row *r=&saved[i];
  printf("%s{\"slot\":%u,\"name\":",i?",":"",r->slot);json_string(r->name);
  printf(",\"function\":%llu,\"module\":%llu,\"tls_context\":%llu,\"stream\":%llu,\"native_tid\":%llu,\"success\":%llu,\"failure\":%llu,\"last_error\":%llu}",(unsigned long long)r->function,(unsigned long long)r->module,(unsigned long long)r->context,(unsigned long long)r->stream,(unsigned long long)r->tid,(unsigned long long)r->success,(unsigned long long)r->failure,(unsigned long long)r->error);
 }
 printf("]}\n");fflush(stdout);return ferror(stdout)?-1:0;
}
int main(int argc,char **argv) {
    if(argc!=9||strcmp(argv[1],"survey")) {fprintf(stderr,"usage: %s survey PID APP BUILD GAME.exe LIBCUDA SHA256 RUNTIME_DIR\n",argv[0]);return 2;}
    char *end; long parsed=strtol(argv[2],&end,10);
    if(*end||parsed<=1||parsed>INT32_MAX||argv[8][0]!='/'||strstr(argv[8],"..")) return 2;
    struct target t={.pid=(int)parsed,.pidfd=-1,.directory=-1,.library_fd=-1,
        .app=argv[3],.build=argv[4],.game=argv[5],.library_path=argv[6]};
    int result=2,library=-1;
    char path[4096],boot[128];
    t.pidfd=(int)syscall(SYS_pidfd_open,t.pid,0);
    if(t.pidfd<0||identity(t.pid,&t.ticks,&t.namespace_pid)) goto cleanup;
    snprintf(path,sizeof path,"/proc/%d",t.pid);
    struct stat process; if(stat(path,&process)||process.st_uid!=getuid()) goto cleanup;
    if(!live(&t)) goto cleanup;
    if(text_file("/proc/sys/kernel/random/boot_id",boot,sizeof boot)<36) goto cleanup;
    memcpy(t.boot,boot,36);
    library=open(argv[6],O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(library<0||fstat(library,&t.library)||!S_ISREG(t.library.st_mode)||mapped_library(t.pid,&t.library)) goto cleanup;
    if(hash_library(library,argv[7])) goto cleanup;
    struct stat after;if(fstat(library,&after)||!same_file(&after,&t.library))goto cleanup;
    t.library_fd=library;
    if(snprintf(path,sizeof path,"/proc/%d/root%s/zluda-launch-survey",t.pid,argv[8])>=(int)sizeof path) goto cleanup;
    t.directory=open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    struct statfs fs;
    if(t.directory<0||private_object(t.directory,1)||fstatfs(t.directory,&fs)||fs.f_type!=0x01021994) goto cleanup;
    if(select_instance(&t)||!live(&t)||!library_unchanged(&t)) goto cleanup;
    signal(SIGINT,stop);signal(SIGTERM,stop);signal(SIGPIPE,SIG_IGN);
    fprintf(stderr,"survey reader ready pid=%d start=%llu build=%s app=%s\n",t.pid,(unsigned long long)t.ticks,t.build,t.app);
    result=0;
    while(!stopping) {
        if(!live(&t))break;
        if(atomic_load_explicit(&t.header->active,memory_order_acquire)!=1)break;
        if(select_instance(&t)||!library_unchanged(&t)) {
            // Destructor can unlink between our first active check and enumeration.
            if(!live(&t)||atomic_load_explicit(&t.header->active,memory_order_acquire)!=1)break;
            result=2;break;
        }
        int sampled=snapshot(&t);
        if(sampled) {if(sampled<0)result=2;break;}
        struct pollfd p={t.pidfd,POLLIN,0};
        int waited=poll(&p,1,1000);
        if(waited<0&&errno!=EINTR) {result=2;break;}
        if(waited>0)break;
    }
cleanup:
    if(result)fprintf(stderr,"recording reader: unavailable (identity/provenance/setup rejected)\n");
    if(library>=0)close(library);
    if(t.header)munmap(t.header,FILE_BYTES);
    if(t.directory>=0)close(t.directory);
    if(t.pidfd>=0)close(t.pidfd);
    return result;
}
