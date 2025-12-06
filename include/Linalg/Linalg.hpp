#pragma once

#include "Linalg/LinalgNaive.hpp"

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
            
            static void set_backend(Backend b);
            static Backend get_backend();

            static Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');
            static Dataframe multiply(const Dataframe& df1, Dataframe& df2);
            static Dataframe transpose(Dataframe& df);
            static Dataframe inverse(Dataframe& df);
            
    };
}; 