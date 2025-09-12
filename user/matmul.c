/*
 * DISTRIBUTED MATRIX MULTIPLICATION PROGRAM
 * 
 * This program demonstrates parallel computing by dividing matrix multiplication
 * work among multiple child processes that communicate via pipes.
 * 
 * CONCEPT: Instead of one process doing all the work, we create multiple child
 * processes where each child computes a portion of the result matrix and sends
 * its results back to the parent process through pipes.
 * 
 * Child 0 computing rows 0 to 2
 *  Child 1 computing rows 3 to 5
 *  Child 2 computing rows 6 to 7
 *  Child 3 computing rows 8 to 9
 * 
 * MATRIX MULTIPLICATION: C = A × B
 * Where C[i][j] = sum of (A[i][k] * B[k][j]) for all k from 0 to N-1
 */

#include "kernel/types.h"
#include "user/user.h" //Provides system call interfaces like fork(), pipe()

#define N 10     // matrix size (NxN)
#define P 4      // Number of child processes to distribute work among

int A[N][N];     // First input matrix
int B[N][N];     // Second input matrix  
int C[N][N];     // Result matrix (computed by distributed processes)
int C_ref[N][N]; // Reference result (computed by single process for verification)


// * FUNCTION: init_matrices()
// * PURPOSE: Initialize matrices A and B with test values

void init_matrices() {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = i + j + 1;       // Example: A[0][0]=1, A[0][1]=2, A[1][0]=2, A[1][1]=3
      B[i][j] = (i == j) ? 2 : 1;
    }
  }
}

// * FUNCTION: multiply_reference()
// * PURPOSE: Compute matrix multiplication using single process (reference implementation)


void multiply_reference() {
  int i, j, k;
  for (i = 0; i < N; i++) {         // for each row
    for (j = 0; j < N; j++) {       // for each column
      int sum = 0;                  // sum initialisation
      for (k = 0; k < N; k++) {     // Sum over all elements in the row/column
        sum += A[i][k] * B[k][j];   // Multiply corresponding elements and add to sum
      }
      C_ref[i][j] = sum;    // Store computed sum in reference result matrix
    }
  }
}

void print_matrix(int M[N][N]) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      printf("%d ", M[i][j]);
    }
    printf("\n");
  }
}

/*
 * FUNCTION: compare_results()
 * PURPOSE: Compare distributed result (C) with reference result (C_ref)
 * RETURNS: 0 if matrices are identical, 1 if they differ
 * 
 * This function verifies that our parallel computation produced the correct result
 */

int compare_results() {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (C[i][j] != C_ref[i][j]) {
        return 1;
      }
    }
  }
  return 0;
}

/*
 * FUNCTION: get_work_range()
 * PURPOSE: Calculate which rows each process should compute (load balancing)
 * INPUT: process_id - which child process (0 to P-1)
 * OUTPUT: start, end - row range for this process to compute
 * 
 * PROBLEM: With N=10 rows and P=4 processes, we have 10/4 = 2 rows per process
 * with 2 remainder rows. Simple division would give processes 0,1,2 each 2 rows
 * and process 3 gets 4 rows (unfair!).
 * 
 * SOLUTION: Distribute remainder evenly. First (N%P) processes get one extra row.
 * Result: processes 0,1 get 3 rows each, processes 2,3 get 2 rows each.
 */

// Better load balancing: distribute rows more evenly
void get_work_range(int process_id, int *start, int *end) {
  int base_rows = N / P;
  int extra_rows = N % P;
  
  if (process_id < extra_rows) {
    *start = process_id * (base_rows + 1);
    *end = *start + base_rows + 1;
  } else {
    *start = extra_rows * (base_rows + 1) + (process_id - extra_rows) * base_rows;
    *end = *start + base_rows;
  }
}

/*
 * MAIN FUNCTION: The heart of our distributed matrix multiplication
 * 
 * ALGORITHM:
 * 1. Initialize matrices and compute reference result
 * 2. Create pipes for parent-child communication
 * 3. Fork P child processes
 * 4. Each child computes its assigned rows and sends results via pipe
 * 5. Parent collects results from all children
 * 6. Compare distributed result with reference result
 */


int main(int argc, char *argv[]) {
  int pipes[P][2];          // Array of P pipes, each pipe has 2 file descriptors [read_end, write_end]
  int pids[P];              // Array to store process IDs of child processes
  int i;                    // Loop counter

  printf("Distributed Matrix Multiplication (%dx%d) with %d processes\n", N, N, P);
  
  // STEP 1: Prepare the data
  init_matrices();
  multiply_reference();

  // STEP 2: Create communication channels (pipes)
  // Each child will have its own pipe to send results back to parent
  for (i = 0; i < P; i++) {      
    if (pipe(pipes[i]) < 0) {       // Create pipe i
      printf("pipe failed for child %d\n", i);
      exit(1);
    }
    // After successful pipe(), pipes[i][0] is read end, pipes[i][1] is write end
  }

  // STEP 3: Create child processes
  for (i = 0; i < P; i++) {
    pids[i] = fork();
    if (pids[i] < 0) {
      printf("fork failed for child %d\n", i);
      exit(1);
    }
    
    if (pids[i] == 0) {
      // Determine which rows this child should compute
      int start, end;
      get_work_range(i, &start, &end);
      
      // Close the read end of our pipe (child only writes, doesn't read)
      close(pipes[i][0]); // Close read end
      
      int out_count = (end - start) * N;
      int outbuf[out_count];
      int idx = 0;
      
      for (int r = start; r < end; r++) {
        for (int c = 0; c < N; c++) {
          int sum = 0;
          for (int k = 0; k < N; k++) {
            sum += A[r][k] * B[k][c];
          }
          outbuf[idx++] = sum;
        }
      }
      
      int bytes = sizeof(int) * out_count;
      int written = 0;
      
      while (written < bytes) {
        int w = write(pipes[i][1], ((char*)outbuf) + written, bytes - written);
        if (w <= 0) {
          printf("Child %d: write failed\n", i);
          break;
        }
        written += w;
      }
      // Clean up and exit child process
      close(pipes[i][1]);
      exit(0);
    }
  }

  // STEP 4: PARENT PROCESS - Collect results from all children
  for (i = 0; i < P; i++) {
    // Determine which rows child i computed
    int start, end;
    get_work_range(i, &start, &end);
    
    int in_count = (end - start) * N;
    int bytes = sizeof(int) * in_count;
    int inbuf[in_count];
    
    close(pipes[i][1]); // Close write end of pipe (parent only reads, doesn't write)
    
    // READ RESULTS: Read all data from child via pipe

    int offset = 0;
    while (offset < bytes) {
      int r = read(pipes[i][0], ((char*)inbuf) + offset, bytes - offset);
      if (r <= 0) {
        printf("Parent: read from child %d failed\n", i);
        break;
      }
      offset += r;
    }
    
    // Copy results into C matrix
    int idx = 0;
    for (int r = start; r < end; r++) {
      for (int c = 0; c < N; c++) {
        C[r][c] = inbuf[idx++];
      }
    }
    
    close(pipes[i][0]);
  }

  // STEP 5: Wait for all child processes to finish
  // This ensures all children have completed before parent continues
  for (i = 0; i < P; i++) {
    wait(0);
  }

  // STEP 6: Display and verify results
  printf("\nResult matrix C (distributed):\n");
  print_matrix(C);
  
  printf("\nReference matrix C_ref (single-threaded):\n");
  print_matrix(C_ref);
  
  if (compare_results() == 0) {
    printf("\nSUCCESS: distributed result matches reference.\n");
  } else {
    printf("\nERROR: distributed result differs from reference.\n");
  }

  exit(0);
}