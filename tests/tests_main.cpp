#include "Utils/TestSuite.hpp"

void tests_data();
void tests_naive();

#ifdef __AVX2__
    void tests_avx2();
#endif

#ifdef USE_MKL
    void tests_mkl();
#endif

void tests_eigen();

int main() {
    tests_data();
    tests_naive();

    #ifdef __AVX2__
        tests_avx2();
    #endif

    tests_eigen();

    #ifdef USE_MKL
        tests_mkl();
    #endif
}