#include "Utils/TestSuite.hpp"
#include <iostream>

namespace TestSuite {

void Tests::run_all() {

    for (auto& f : to_test) {
        try {
            f.first();

            std::cout << "✓ " << f.second << std::endl;
            success++;
        }
        catch (const std::exception& e) {
            std::cerr << "✗ " << f.second << ": " << e.what() << std::endl;
            failure++;
        }
    }
    std::cout << "\n" << success << "/" << success+failure << "\n" << std::endl;
}
}