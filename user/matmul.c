#include "kernel/types.h"
#include "user/user.h"

#define MAX 400   // you can try 100, 200, 400 for experiments

int A[MAX][MAX];
int B[MAX][MAX];
int C[MAX][MAX];

void fill_matrix(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = i + j;
      B[i][j] = i - j;
      C[i][j] = 0;
    }
  }
}

void matmul(int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main() {
  int sizes[] = {100, 200, 400};  // test different matrix sizes
  int num_sizes = 3;
  
  printf("=== MATRIX MULTIPLICATION BENCHMARK ===\n");
  printf("Matrix_Size,Cycles,Time,Instructions\n");  // CSV header for easy parsing
  
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