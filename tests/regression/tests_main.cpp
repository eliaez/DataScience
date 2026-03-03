#include "TestSuite.hpp"
#include <iostream>
#include <string>

void tests_OLS();
void tests_Ridge();

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_name>\n";
        return 1;
    }
    
    std::string test = argv[1];
    
    if (test == "OLS") {
        tests_OLS();
    } 
    else if (test == "Ridge") {
        tests_Ridge();
    }
    else {
        std::cerr << "Test inconnu: " << test << "\n";
        return 1;
    }
    return 0;
}