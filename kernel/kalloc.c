// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"

#ifndef LAB_PGTBL
#define SUPERPGSIZE (2 * (1 << 20))
#define SUPERPGROUNDUP(sz) (((sz)+SUPERPGSIZE-1) & ~(SUPERPGSIZE-1))
#endif

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

// Superpage allocator
#define NSUPERPAGES 8  // Number of 2MB superpages to reserve
struct {
  struct spinlock lock;
  char *superpages[NSUPERPAGES];  // Array of 2MB regions
  int used[NSUPERPAGES];          // 1 if used, 0 if free
} superkmem;

void
kinit()
{
  initlock(&kmem.lock, "kmem");
  initlock(&superkmem.lock, "superkmem");
  
  // Initialize superpage allocator
  for(int i = 0; i < NSUPERPAGES; i++) {
    superkmem.superpages[i] = 0;
    superkmem.used[i] = 0;
  }
  
  // Reserve some 2MB-aligned regions for superpages before general allocation
  char *p = (char*)PGROUNDUP((uint64)end);
  
  // Find 2MB-aligned regions and reserve them
  int reserved = 0;
  while(p + SUPERPGSIZE <= (char*)PHYSTOP && reserved < NSUPERPAGES) {
    // Align to 2MB boundary
    uint64 aligned = SUPERPGROUNDUP((uint64)p);
    if(aligned + SUPERPGSIZE <= PHYSTOP) {
      superkmem.superpages[reserved] = (char*)aligned;
      reserved++;
      p = (char*)(aligned + SUPERPGSIZE);
    } else {
      break;
    }
  }
  
  // Free the remaining memory for regular allocation
  freerange(p, (void*)PHYSTOP);
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}

// Free the page of physical memory pointed at by pa,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r)
    kmem.freelist = r->next;
  release(&kmem.lock);

  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}

// Allocate one 2MB superpage of physical memory.
// Returns a 2MB-aligned pointer that the kernel can use.
// Returns 0 if no superpage is available.
void *
superalloc(void)
{
  acquire(&superkmem.lock);
  
  for(int i = 0; i < NSUPERPAGES; i++) {
    if(superkmem.superpages[i] && !superkmem.used[i]) {
      superkmem.used[i] = 1;
      void *result = superkmem.superpages[i];
      release(&superkmem.lock);
      
      // Clear the superpage
      memset(result, 0, SUPERPGSIZE);
      return result;
    }
  }
  
  release(&superkmem.lock);
  return 0;  // No superpage available
}

// Free a 2MB superpage.
void
superfree(void *pa)
{
  if(((uint64)pa % SUPERPGSIZE) != 0)
    panic("superfree: not superpage aligned");
    
  acquire(&superkmem.lock);
  
  for(int i = 0; i < NSUPERPAGES; i++) {
    if(superkmem.superpages[i] == pa) {
      if(!superkmem.used[i])
        panic("superfree: already free");
      superkmem.used[i] = 0;
      release(&superkmem.lock);
      return;
    }
  }
  
  release(&superkmem.lock);
  panic("superfree: not a superpage");
}