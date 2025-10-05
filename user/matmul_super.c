#include "kernel/types.h"
#include "user/user.h"

#define MAX 400

#define SUPERPGROUNDUP(addr) (((addr) + (2*1024*1024) - 1) & ~((2*1024*1024) - 1))


// Pointers instead of static arrays
int *A;
int *B;
int *C;

void allocate_matrices(int n) {
  // First, align to 2MB boundary
  char *current = sbrk(0);  // Get current break
  uint64 current_addr = (uint64)current;
  uint64 aligned_addr = SUPERPGROUNDUP(current_addr);
  uint64 padding = aligned_addr - current_addr;
  
  if(padding > 0) {
    sbrk(padding);  // Align to 2MB
  }
  
  // Now allocate the matrices (they'll be 2MB-aligned)
  uint64 size_needed = 3 * MAX * MAX * sizeof(int);
  char *p = sbrk(size_needed);
  if(p == (char*)-1) {
    printf("sbrk failed\n");
    exit(1);
  }
  
  printf("Allocated at address: %p\n", p);
  printf("Is A 2MB-aligned? %s\n", 
         ((uint64)p % (2*1024*1024) == 0) ? "YES" : "NO");
  
  A = (int*)p;
  B = (int*)(p + MAX * MAX * sizeof(int));
  C = (int*)(p + 2 * MAX * MAX * sizeof(int));
}

void fill_matrix(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i * MAX + j] = i + j;
      B[i * MAX + j] = i - j;
      C[i * MAX + j] = 0;
    }
  }
}

void matmul(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i * MAX + j] += A[i * MAX + k] * B[k * MAX + j];
      }
    }
  }
}

int main() {
  int sizes[] = {100, 200, 400};
  int num_sizes = 3;
  
  printf("=== MATRIX MULTIPLICATION BENCHMARK ===\n");
  printf("Matrix_Size,Cycles,Time,Instructions\n");
  
  // Allocate matrices once with sbrk
  allocate_matrices(MAX);
  
  for (int s = 0; s < num_sizes; s++) {
    int n = sizes[s];
    
    printf("Testing matrix size %dx%d...\n", n, n);
    fill_matrix(n);
    
    uint64 c1 = rdcycle();
    uint64 t1 = rdtime();
    uint64 i1 = rdinstret();
    
    matmul(n);
    
    uint64 c2 = rdcycle();
    uint64 t2 = rdtime();
    uint64 i2 = rdinstret();
    
    printf("%d,%lu,%lu,%lu\n", n, c2-c1, t2-t1, i2-i1);
    printf("  Cycles: %lu\n", c2 - c1);
    printf("  Time: %lu\n", t2 - t1);
    printf("  Instructions: %lu\n", i2 - i1);
    printf("\n");
  }
  
  printf("=== BENCHMARK COMPLETE ===\n");
  exit(0);
}