#include "Linalg/LinalgMKL.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace MKL {
#ifdef USE_MKL

Dataframe transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    // Changing layout for better performances later
    if (df.get_storage()){
        df.change_layout_inplace("MKL");
    }

    std::vector<double> data = Dataframe::transpose_mkl(temp_row, temp_col, df.get_data());

    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, 
    size_t m, size_t n, char op = '+') { 

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
            -1.0,           // Scalar beta
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
        v1.data(),       // Input df1
        m,                  
        v2.data(),       // Input df2
        o,                  
        0.0,                // Scalar beta : res = alpha × op(df1) × op(df2) + beta × res
        new_data.data(),    // Output res
        m                   
    );

    // Return column - major
    return new_data;
}

Dataframe inverse(Dataframe& df) {

    size_t n = df.get_cols();
    std::vector<double> new_data = df.get_data();
    std::vector<lapack_int> swaps(n);

    lapack_int msg = LAPACKE_dgetrf(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,
        n,         
        new_data.data(),    // Matrix (input: data, output: LU)
        n,      
        swaps.data()        // Pivots
    );
    if(msg != 0) throw std::runtime_error("LU factorization failed");

    lapack_int msg1 = LAPACKE_dgetri(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,       
        new_data.data(),    // Matrix (input: LU, output: inverse)
        n,      
        swaps.data()        // Pivots
    );
    if(msg1 != 0) throw std::runtime_error("Matrix inversion failed");

    return Dataframe(n, n, false, std::move(new_data));
}

#endif
}
}