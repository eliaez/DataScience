#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace AVX2 {

        constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        constexpr size_t PREFETCH_DIST = 64; // Pre-fetch 64*8 bytes ahead 
        
        // Function to test if the data from df is a triangular matrix, 
        // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
        int triangular_matrix(Dataframe& df);

        Dataframe sum(Dataframe& df1, Dataframe& df2, char op = '+');
        Dataframe multiply(Dataframe& df1, Dataframe& df2);
        Dataframe transpose(Dataframe& df);

        // Function to calculate determinant of Matrix from Df data, through either the product of 
        // the diagonal if the matrix is triangular or with LU decomposition. 
        // Returns determinant, LU matrix 
        std::tuple<double, std::vector<double>, Dataframe> determinant(Dataframe& df);

        // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
        std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(Dataframe& df);

        // Function to solve LU system with Forward substitution and Backward substitution
        Dataframe solveLU_inplace(Dataframe& perm, Dataframe& LU);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
    }
}