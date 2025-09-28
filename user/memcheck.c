#include "kernel/types.h"
#include "user/user.h"

int main(int argc, char *argv[]) {
    printf("Memory check program\n");

    // Just print PHYSTOP value (from kernel)
    // since user space can't directly access memlayout.h
    // we simulate by allocating memory until failure.
    
    int step = 1024 * 1024; // 1 MB
    int total = 0;

    while (1) {
        char *p = malloc(step);
        if (p == 0) {
            break; // out of memory
        }
        total += step;
        printf("Allocated %d MB\n", total / (1024*1024));
    }

    printf("Total allocated before failure: %d MB\n", total / (1024*1024));
    exit(0);
}
