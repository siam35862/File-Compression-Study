#include <iostream>
#include "paq7.h"

int main() {
    // 1. Create mutable string arguments
    char arg0[] = "paq7";
    char arg1[] = "-3";
    char arg2[] = "my_archive.paq";
    char arg3[] = "decomp.txt";

    // 2. Build the argv array
    char* argv[] = {arg0,  arg2, arg3};
    int argc = 3;

    std::cout << "Starting compression from another file..." << std::endl;

    // 3. Call the renamed function
    int result = paq7_main(argc, argv);

    std::cout << "Process finished with exit code: " << result << std::endl;
    
    return 0;
}