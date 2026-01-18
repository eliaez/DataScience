#pragma once

#include "Linalg/LinalgNaive.hpp"
#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/LinalgAVX2_threaded.hpp"
#include "Linalg/LinalgEigen.hpp"
#include "Linalg/LinalgMKL.hpp"
#include <string>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

namespace Linalg {

    #if defined(__AVX2__) && defined(USE_MKL)
        enum class Backend {
            NAIVE,
            AVX2,
            AVX2_THREADED,
            EIGEN,
            MKL,
            AUTO
        };
    #elif defined(__AVX2__)
        enum class Backend {
            NAIVE,
            AVX2,
            AVX2_THREADED,
            EIGEN,
            AUTO
        };
    #elif defined(USE_MKL)
        enum class Backend {
            NAIVE,
            EIGEN,
            MKL,
            AUTO
        };
    #else 
        enum class Backend {
            NAIVE, // Still col-major and cache friendly
            EIGEN,
            AUTO
        };
    #endif

    class Operations {

        private:
            static Backend current_backend;

        public:
            
            // Setter 
            static void set_backend(Backend b);
            static void set_backend(const std::string& b);
            
            // Will select the best one if AUTO
            static Backend get_backend();

            static Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');
            static Dataframe multiply(const Dataframe& df1, const Dataframe& df2);
            static Dataframe transpose(Dataframe& df);
            static Dataframe inverse(Dataframe& df);
            
    };

    // Function to get Backend in string
    std::string get_backend();

    // Function to test if the data from df is a triangular matrix, 
    // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
    int triangular_matrix(const Dataframe& df);

    #ifdef __AVX2__
        // Function to test if the data from df is a triangular matrix, 
        // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
        int triangular_matrix_avx2(const Dataframe& df);
    #endif

    // Function to calculate determinant of Matrix from Df data, through either the product of 
    // the diagonal if the matrix is triangular or with LU decomposition. 
    // Returns determinant, LU matrix 
    std::tuple<double, std::vector<double>, std::vector<double>> determinant(Dataframe& df);
}; 