#pragma once

#include <vector>

namespace Linalg::detail {
    class OperationsImpl {
        public:
            static std::vector<double> sum_impl(
                const std::vector<double>& v1,
                const std::vector<double>& v2, 
                size_t v1_rows,
                size_t v1_cols,
                size_t v2_rows,
                size_t v2_cols,
                bool v1_layout,
                bool v2_layout,
                char op = '+'
            );
        
            static std::vector<double> multiply_impl(
                const std::vector<double>& v1,
                const std::vector<double>& v2, 
                size_t v1_rows,
                size_t v1_cols,
                size_t v2_rows,
                size_t v2_cols,
                bool v1_layout,
                bool v2_layout
            );

            static std::vector<double> transpose_impl(
                const std::vector<double>& v1, 
                size_t v1_rows,
                size_t v1_cols,
                bool v1_layout
            );

            static std::vector<double> inverse_impl(
                const std::vector<double>& v1, 
                size_t v1_rows,
                size_t v1_cols,
                bool v1_layout
            );

            static int triangular_impl(
                const std::vector<double>& v1, 
                size_t v1_rows,
                size_t v1_cols,
                bool v1_layout
            );

            #ifdef __AVX2__
                static int triangular_avx2_impl(
                    const std::vector<double>& v1, 
                    size_t v1_rows,
                    size_t v1_cols,
                    bool v1_layout
                );
            #endif

            static std::tuple<double, std::vector<double>, std::vector<double>> determinant(
                const std::vector<double>& v1, 
                size_t v1_rows,
                size_t v1_cols,
                bool v1_layout
            );
    };
}