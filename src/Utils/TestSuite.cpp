#include "TestSuite.hpp"
#include <iostream>

namespace TestSuite {

void Tests::run_all() {

    for (auto& f : to_test) {
        auto case_ = f.second.find("Error"); 
        try {
            f.first();

            if (case_ == std::string_view::npos) {
                std::cout << "✓ " << f.second << std::endl;
                success++;
            }
            else {
                std::cerr << "✗ " << f.second << ": Not expected result, " << std::endl;
                failure++;
            }
            
        }
        catch (const std::exception& e) {
            if (case_ != std::string_view::npos) {
                std::cerr << "✓ " << f.second << ": " << e.what() << std::endl;
                success++;
            }
            else {
                std::cerr << "✗ " << f.second << ": Not expected result, " << e.what() << std::endl;
                failure++;
            }
        }
    }
    std::cout << "\n" << success << "/" << (success+failure) << "\n" << std::endl;
}
}