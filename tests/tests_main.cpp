#include "Utils/TestSuite.hpp"

void tests_data();
void tests_naive();
void tests_AVX2();

int main() {
    tests_data();
    tests_naive();
    tests_AVX2();
}