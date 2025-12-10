#include "Utils/TestSuite.hpp"

void tests_data();
void tests_naive();

#ifdef __AVX2__
void tests_AVX2();
#endif

int main() {
    tests_data();
    tests_naive();

    #ifdef __AVX2__
    tests_AVX2();
    #endif
}