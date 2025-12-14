#pragma once

#include <vector>
#include <functional>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace TestSuite {

    class Tests {
        private:
            int success;
            int failure;
            std::vector<std::pair<std::function<void()>,std::string_view>> to_test;

        public:
            Tests() = default;

            const std::vector<std::pair<std::function<void()>,std::string_view>>& get_testlist() const { return to_test; }

            void add_test(std::function<void()> f, std::string_view s) { to_test.emplace_back(std::move(f), std::move(s)); }

            void run_all();
    };
};

// Macro Assert equal function
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

#define ASSERT_EQ_INT(actual, expected) \
    do { \
        auto _actual = static_cast<int>(actual); \
        auto _expected = static_cast<int>(expected); \
        if (_actual != _expected) { \
            throw std::runtime_error( \
                std::string("Ligne ") + std::to_string(__LINE__) \
            ); \
        } \
    } while(0);

#define ASSERT_EQ_SCI3(actual, expected) \
    do { \
        auto _actual = (actual); \
        auto _expected = (expected); \
        std::ostringstream oss_a, oss_e; \
        oss_a << std::scientific << std::setprecision(3) << _actual; \
        oss_e << std::scientific << std::setprecision(3) << _expected; \
        if (oss_a.str() != oss_e.str()) { \
            throw std::runtime_error( \
                std::string("Ligne ") + std::to_string(__LINE__) \
            ); \
        } \
    } while(0);
    
