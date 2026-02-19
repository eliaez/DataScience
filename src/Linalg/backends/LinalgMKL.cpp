#include "Data/Data.hpp"
#include "LinalgMKL.hpp"

#ifdef USE_MKL
    #include <mkl.h>
#endif

namespace Linalg::MKL {
#ifdef USE_MKL

std::vector<double> transpose(const std::vector<double>& v1,  
    size_t v1_rows, size_t v1_cols) {

    std::vector<double> new_data = Dataframe::transpose_mkl(v1_rows, v1_cols, v1);

    return new_data;
}

std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, 
    size_t m, size_t n, char op) { 

    // New data
    std::vector<double> new_data(m * n);

    if (op == '+') {
        mkl_domatadd(
            'C',           // Col-major
            'N',           // Without Transpo
            'N',           // Without Transpo
            m, n,          
            1.0,           // Scalar alpha
            v1.data(), m,
            1.0,           // Scalar beta
            v2.data(), m,   
            new_data.data(), m    
        );
    }
    else if (op == '-') {
        mkl_domatadd(
            'C',           // Col-major
            'N',           // Without Transpo
            'N',           // Without Transpo
            m, n,          
            1.0,           // Scalar alpha
            v1.data(), m,
            -1.0,          // Scalar beta
            v2.data(), m,   
            new_data.data(), m    
        );
    }

    return new_data;
}

std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
    size_t m, size_t n, size_t o, size_t p ) {

    std::vector<double> new_data(m * p);

    cblas_dgemm(
        CblasColMajor,      // CblasRowMajor or CblasColMajor
        CblasNoTrans,       // CblasNoTrans, CblasTrans or CblasConjTrans for df1
        CblasNoTrans,       // CblasNoTrans, CblasTrans or CblasConjTrans for df2
        m,                
        p,                
        n,                
        1.0,                // Scalar alpha : res = alpha × op(df1) × op(df2) + beta × res
        v1.data(),          // Input df1
        m,                  
        v2.data(),          // Input df2
        o,                  
        0.0,                // Scalar beta : res = alpha × op(df1) × op(df2) + beta × res
        new_data.data(),    // Output res
        m                   
    );

    // Return column - major
    return new_data;
}

std::vector<double> inverse(const std::vector<double>& v1, size_t n,
    std::vector<double>, std::vector<double>) {

    std::vector<double> new_data = v1;
    std::vector<lapack_int> pivots(n);

    lapack_int msg = LAPACKE_dgetrf(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,
        n,         
        new_data.data(),    // Matrix (input: data, output: LU)
        n,      
        pivots.data()        // Pivots
    );
    if(msg != 0) throw std::runtime_error("LU factorization failed");

    lapack_int msg1 = LAPACKE_dgetri(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,       
        new_data.data(),    // Matrix (input: LU, output: inverse)
        n,      
        pivots.data()        // Pivots
    );
    if(msg1 != 0) throw std::runtime_error("Matrix inversion failed");

    return new_data;
}

#endif
}