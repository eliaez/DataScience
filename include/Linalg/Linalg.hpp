#pragma once

#include "Data/Data.hpp"
#include <string>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

namespace Linalg {

    #if defined(__AVX2__) && defined(USE_MKL)
        enum class Backend { NAIVE, AVX2, AVX2_THREADED, EIGEN, MKL, AUTO };
    #elif defined(__AVX2__)
        enum class Backend { NAIVE, AVX2, AVX2_THREADED, EIGEN, AUTO };
    #elif defined(USE_MKL)
        enum class Backend { NAIVE, EIGEN, MKL, AUTO };
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
            class Impl; // PIMPL - implementation details hidden

        public:
            
            // Setter 
            static void set_backend(Backend b);
            static void set_backend(const std::string& b);
            
            // Will select the best one (implemented) if AUTO
            static Backend get_backend();

            // Sum function between two dataframes, will choose the backend accordingly to your config.
            // op stands for operator (+ or -), by default '+' 
            // For Naive, AVX2, AVX2TH, layout need to be row - row or col - col 
            // For Eigena and MKL, layout need to be col - col
            static Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');

            // Mult function between two dataframes, will choose the backend accordingly to your config.
            // For Naive, AVX2, AVX2TH, layout need to be row - col
            // For Eigena and MKL, layout need to be col - col
            static Dataframe multiply(const Dataframe& df1, const Dataframe& df2);

            // Transpose function, will choose the backend accordingly to your config.
            // If your dataframe layout is row major then it will change it to col major by default.
            static Dataframe transpose(Dataframe& df);

            // Inv function, will choose the backend accordingly to your config.
            // Layout need to be col major for performance purpose. 
            static Dataframe inverse(Dataframe& df);

            // Function to calculate determinant of Matrix from Df data, through either the product of 
            // the diagonal if the matrix is triangular or with LU decomposition. 
            // Returns determinant, permutation matrix and LU matrix 
            static std::tuple<double, std::vector<double>, std::vector<double>> determinant(Dataframe& df);

            // Function to test if the data from df is a triangular matrix, 
            // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
            static int triangular_matrix(const Dataframe& df);

            #ifdef __AVX2__
                // Function to test if the data from df is a triangular matrix, 
                // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
                static int triangular_matrix_avx2(const Dataframe& df);
            #endif
    };

    // Function to get Backend in string
    std::string get_backend();
}; 