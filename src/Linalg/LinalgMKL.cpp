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

Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();

    // Verify if we can sum them
    if (m != o || n != p) throw std::runtime_error("Need two Matrix of equal dimensions");

    // Condition to have better performances
    if ((df1.get_storage() != df2.get_storage()) && df1.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage and Col-major for performances purpose");
    }

    // New data
    std::vector<double> new_data(m * n);

    if (op == '+') {
        mkl_domatadd(
            'C',           // Col-major
            'N',           // Without Transpo
            'N',           // Without Transpo
            m, n,          
            1.0,           // Scalar alpha
            df1.get_db(), m,
            1.0,           // Scalar beta
            df2.get_db(), m,   
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
            df1.get_db(), m,
            -1.0,           // Scalar beta
            df2.get_db(), m,   
            new_data.data(), m    
        );
    }

    return {m, n, false, std::move(new_data)};
}

Dataframe multiply(const Dataframe& df1, const Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    if (n != o) throw std::runtime_error("Need df1 cols == df2 rows");

    // Condition to have better performances
    if ((df1.get_storage() != df2.get_storage()) && df1.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage and Col-major for performances purpose");
    }

    std::vector<double> new_data(m * p);

    cblas_dgemm(
        CblasColMajor,      // CblasRowMajor or CblasColMajor
        CblasNoTrans,       // CblasNoTrans, CblasTrans or CblasConjTrans for df1
        CblasNoTrans,       // CblasNoTrans, CblasTrans or CblasConjTrans for df2
        m,                
        p,                
        n,                
        1.0,                // Scalar alpha : res = alpha × op(df1) × op(df2) + beta × res
        df1.get_db(),       // Input df1
        m,                  
        df2.get_db(),       // Input df2
        o,                  
        0.0,                // Scalar beta : res = alpha × op(df1) × op(df2) + beta × res
        new_data.data(),    // Output res
        m                   
    );

    // Return column - major
    return Dataframe(m, p, false, std::move(new_data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}

Dataframe inverse(Dataframe& df) {

    size_t n = df.get_cols();
    std::vector<double> new_data(n * n) = df.get_data();
    std::vector<int> swaps(n);

    int msg = LAPACKE_dgetrf(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,         
        new_data.data(),       // Matrix (input: data, output: LU)
        n,      
        swaps.data()    // Pivots
    );

    if(msg != 0) throw std::runtime_error("Matrice singulière ou erreur");

    int msg1 = LAPACKE_dgetri(
        LAPACK_COL_MAJOR,   // LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
        n,         
        new_data.data(),       // Matrix (input: LU, output: inverse)
        n,      
        swaps.data()    // Pivots
    );

    return Dataframe(n, n, false, std::move(new_data));
}

#endif
}
}