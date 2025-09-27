#include "kernel/types.h" 
#include "user/user.h" 

int main() { 
    uint64 c1 = rdcycle(); 
    uint64 t1 = rdtime(); 
    uint64 i1 = rdinstret(); 
    
    // Run some dummy loop 
    for (int i = 0; i < 1000000; i++); 
    
    uint64 c2 = rdcycle(); 
    uint64 t2 = rdtime(); 
    uint64 i2 = rdinstret(); 
    
    printf("Cycles: %lu\n", c2 - c1);
    printf("Time: %lu\n", t2 - t1); 
    printf("Instructions: %lu\n", i2 - i1); 
    
    exit(0); 
}