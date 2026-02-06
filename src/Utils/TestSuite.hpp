#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <functional>

namespace TestSuite {

    class Tests {
        private:
            int success = 0;
            int failure = 0;
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

// Macro to compare two vectors of the same type with espilon = 1e-9 as lower threshold
#define ASSERT_EQ_VEC_EPS(actual, expected, EPSILON) \
    do { \
        auto _actual = (actual); \
        auto _expected = (expected); \
        for (size_t i = 0; i < _actual.size(); i++) { \
            if (std::abs(_actual[i] - _expected[i]) > EPSILON) { \
                throw std::runtime_error( \
                    std::string("Ligne ") + std::to_string(__LINE__) \
                ); \
            } \
        } \
    } while(0); 

// Macro to compare two vectors of the same type with relative espilon (%) as lower threshold
#define ASSERT_VEC_EPS(actual, expected, EPSILON) \
    do { \
        auto _actual = (actual); \
        auto _expected = (expected); \
        for (size_t i = 0; i < _actual.size(); i++) { \
            if (std::abs(_actual[i] - _expected[i]) / std::max(std::abs(_actual[i]), std::abs(_expected[i])) > EPSILON) { \
                throw std::runtime_error( \
                    std::string("Ligne ") + std::to_string(__LINE__) \
                ); \
            } \
        } \
    } while(0); 
    
