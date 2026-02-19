#include "TestSuite.hpp"
#include <iostream>
#include <string>

void tests_data();
void tests_naive();

#ifdef __AVX2__
    void tests_avx2();
    void tests_avx2_th();
#endif

#ifdef USE_MKL
    void tests_mkl();
#endif

void tests_eigen();

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_name>\n";
        return 1;
    }
    
    std::string test = argv[1];
    
    if (test == "data") {
        tests_data();
    } else if (test == "naive") {
        tests_naive();
    } else if (test == "avx2") {
        #ifdef __AVX2__
            tests_avx2();
        #endif
    } else if (test == "avx2_th") {
        #ifdef __AVX2__
            tests_avx2_th();
        #endif
    } else if (test == "eigen") {
        tests_eigen();
    } else if (test == "mkl") {
        #ifdef USE_MKL
            tests_mkl();
        #endif
    } else {
        std::cerr << "Test inconnu: " << test << "\n";
        return 1;
    }
    
    return 0;
}