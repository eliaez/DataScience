#pragma once

#include "Linalg/LinalgNaive.hpp"
#include "Linalg/LinalgAVX2.hpp"

namespace Linalg {

    enum class Backend {
        NAIVE, // Still col-major and cache friendly
        AVX2,
        AVX2_THREADED,
        EIGEN,
        AUTO
    };

    class Operations {

        private:
            static Backend current_backend;

        public:
            
            // Setter 
            static void set_backend(Backend b);
            
            // Will select the best one if AUTO
            static Backend get_backend();

            static Dataframe sum(Dataframe& df1, Dataframe& df2, char op = '+');
            static Dataframe multiply(Dataframe& df1, Dataframe& df2);
            static Dataframe transpose(Dataframe& df);
            static Dataframe inverse(Dataframe& df);
            
    };

    // Function to test if the data from df is a triangular matrix, 
    // 3 for diagonal, 2 for Up, 1 for Down and 0 if not.
    int triangular_matrix(const Dataframe& df);
}; 