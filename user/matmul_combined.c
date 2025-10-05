#include "kernel/types.h"
#include "user/user.h"

#define MAX 400

#ifdef USE_SUPERPAGES
// Superpages version: dynamic allocation
#define SUPERPGROUNDUP(addr) (((addr) + (2*1024*1024) - 1) & ~((2*1024*1024) - 1))

int *A;
int *B;
int *C;

void allocate_matrices(int n) {
  // Align to 2MB boundary
  char *current = sbrk(0);
  uint64 current_addr = (uint64)current;
  uint64 aligned_addr = SUPERPGROUNDUP(current_addr);
  uint64 padding = aligned_addr - current_addr;
  
  if(padding > 0) {
    sbrk(padding);
  }
  
  // Allocate matrices
  uint64 size_needed = 3 * MAX * MAX * sizeof(int);
  char *p = sbrk(size_needed);
  if(p == (char*)-1) {
    printf("sbrk failed\n");
    exit(1);
  }
  
  printf("Using SUPERPAGES: Allocated at %p (2MB-aligned: %s)\n", 
         p, ((uint64)p % (2*1024*1024) == 0) ? "YES" : "NO");
  
  A = (int*)p;
  B = (int*)(p + MAX * MAX * sizeof(int));
  C = (int*)(p + 2 * MAX * MAX * sizeof(int));
}

#define GET_A(i, j) A[(i) * MAX + (j)]
#define GET_B(i, j) B[(i) * MAX + (j)]
#define GET_C(i, j) C[(i) * MAX + (j)]
#define SET_A(i, j, val) A[(i) * MAX + (j)] = (val)
#define SET_B(i, j, val) B[(i) * MAX + (j)] = (val)
#define SET_C(i, j, val) C[(i) * MAX + (j)] = (val)

#else
// Baseline version: static arrays
int A[MAX][MAX];
int B[MAX][MAX];
int C[MAX][MAX];

void allocate_matrices(void) {
  printf("Using BASELINE: Static arrays (4KB pages)\n");
  // No allocation needed for static arrays
}

#define GET_A(i, j) A[i][j]
#define GET_B(i, j) B[i][j]
#define GET_C(i, j) C[i][j]
#define SET_A(i, j, val) A[i][j] = (val)
#define SET_B(i, j, val) B[i][j] = (val)
#define SET_C(i, j, val) C[i][j] = (val)

#endif

void fill_matrix(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      SET_A(i, j, i + j);
      SET_B(i, j, i - j);
      SET_C(i, j, 0);
    }
  }
}

void matmul(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        SET_C(i, j, GET_C(i, j) + GET_A(i, k) * GET_B(k, j));
      }
    }
  }
}

int main() {
  int sizes[] = {100, 200, 400};
  int num_sizes = 3;
  
  printf("=== MATRIX MULTIPLICATION BENCHMARK ===\n");
  
  // Allocate matrices (different method based on compilation flag)
  allocate_matrices();
  
  printf("Matrix_Size,Cycles,Time,Instructions\n");
  
  for (int s = 0; s < num_sizes; s++) {
    int n = sizes[s];
    
    printf("Testing matrix size %dx%d...\n", n, n);
    fill_matrix(n);
    
    // Measure performance
    uint64 c1 = rdcycle();
    uint64 t1 = rdtime();
    uint64 i1 = rdinstret();
    
    matmul(n);
    
    uint64 c2 = rdcycle();
    uint64 t2 = rdtime();
    uint64 i2 = rdinstret();
    
    // Output in CSV format for easy parsing later
    printf("%d,%lu,%lu,%lu\n", n, c2-c1, t2-t1, i2-i1);
    
    // Also output in readable format
    printf("  Cycles: %lu\n", c2 - c1);
    printf("  Time: %lu\n", t2 - t1);
    printf("  Instructions: %lu\n", i2 - i1);
    printf("\n");
  }
  
  printf("=== BENCHMARK COMPLETE ===\n");
  exit(0);
}