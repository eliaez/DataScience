#include "TestSuite.hpp"
#include <iostream>
#include <string>

void tests_OLS();
void tests_Reg();

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_name>\n";
        return 1;
    }
    
    std::string test = argv[1];
    
    if (test == "OLS") {
        tests_OLS();
    } 
    else if (test == "Reg") {
        tests_Reg();
    }
    else {
        std::cerr << "Test inconnu: " << test << "\n";
        return 1;
    }
    return 0;
}