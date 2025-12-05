#pragma once

#include <vector>
#include <functional>
#include <string>
#include <stdexcept>

namespace TestSuite {

    class Tests {
        private:
            int success;
            int failure;
            std::vector<std::pair<std::function<void()>,std::string_view>> to_test;

        public:
            Tests() = default;

            int get_success() const { return success; }
            int get_failure() const { return failure; }
            const std::vector<std::pair<std::function<void()>,std::string_view>>& get_testlist() const { return to_test; }

            void add_test(std::function<void()> f, std::string_view s) { to_test.emplace_back(std::move(f), std::move(s)); }

            void run_all();
    };
};

#define ASSERT_EQ(actual, expected) \
    do { \
        auto _actual = (actual); \
        auto _expected = (expected); \
        if (_actual != _expected) { \
            throw std::runtime_error( \
                std::string("Ligne ") + std::to_string(__LINE__) \
            ); \
        } \
    } while(0); 
    
