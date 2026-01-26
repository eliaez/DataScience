#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <Eigen/Dense>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

#ifdef USE_MKL
    #include <mkl.h>
#endif

#pragma once


class Dataframe
{   
    private:
        size_t rows;
        size_t cols;
        bool is_row_major;
        std::vector<double> data;
        std::vector<std::string> headers;
        std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder;
        std::unordered_set<int> encoded_cols;

    public:
        #ifdef __AVX2__
        static constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        static constexpr size_t PREFETCH_DIST1 = 4; // Pre-fetch 4*64 bytes ahead for Blocks algo
        #endif

    public: 
        // -------------------------Constructor----------------------------------

        // Default - Dataframe constructor
        Dataframe(
            size_t r = 0, size_t c = 0, bool i = true, 
            std::vector<double> d = {}, 
            std::vector<std::string> h = {},
            std::unordered_map<int, std::unordered_map<std::string, int>> l = {}, 
            std::unordered_set<int> e = {}) 
        : rows(r), cols(c), is_row_major(i), 
            data(std::move(d)), 
            headers(std::move(h)), 
            label_encoder(std::move(l)), 
            encoded_cols(std::move(e)) {}
        
        // -------------------------Getters----------------------------------

        // Getting value from Dataframe according to index
        const double& at(size_t idx) const;
        double& at(size_t idx);
        
        size_t get_rows() const { return rows; }
        size_t get_cols() const { return cols; }
        size_t size() const { return data.size(); }
        bool get_storage() const {return is_row_major; } // Get your layout Row or Col major
        const double* get_db() const { return data.data(); }
        const std::vector<double>& get_data() const { return data; }

        const std::vector<std::string>& get_headers() const { return headers; }
        const std::unordered_set<int>& get_encodedCols() const { return encoded_cols; }
        const std::unordered_map<int, std::unordered_map<std::string, int>>& get_encoder() const { return label_encoder; }

        // -------------------------Operators and others----------------------------------
        
        // Getting val(i, j) according to our storage config, for performance purpose use .at(idx) 
        const double& operator()(size_t i, size_t j) const;
        double& operator()(size_t i, size_t j);

        // Getting a copy of col's data from your df
        std::vector<double> operator[](const std::vector<size_t>& cols_idx) const;
        std::vector<double> operator[](const std::vector<std::string>& cols_name) const;
        std::vector<double> operator[](size_t j) const;
        std::vector<double> operator[](const std::string& col_name) const;

        Dataframe operator+(const Dataframe& other) const { return Linalg::Operations::sum(*this, other); }
        Dataframe operator-(const Dataframe& other) const { return Linalg::Operations::sum(*this, other, '-'); }
        Dataframe operator*(const Dataframe& other) const { return Linalg::Operations::multiply(*this, other); }
        Dataframe operator~() { return Linalg::Operations::transpose(*this);}

        // See Linalg.hpp for more details
        Dataframe inv() { return Linalg::Operations::inverse(*this); }
        
        // See Linalg.hpp for more details
        std::tuple<double, std::vector<double>, std::vector<double>> det() { return Linalg::Operations::determinant(*this); }
        
        // See Linalg.hpp for more details
        int is_tri() const { 
            return Linalg::Operations::triangular_matrix(*this); 
        }
    
        // -------------------------Methods----------------------------------

        // Return corresponding label from a value
        std::string decode_label(int value, int col) const;

        // Displaying our datas with encoded values
        void display_raw(size_t nb_rows, int space = 15) const;
        void display_raw() const {display_raw(rows, 15);}

        // Displaying our datas with decoded values
        void display_decoded(size_t nb_rows, int space = 15) const;
        void display_decoded() const {display_decoded(rows, 15);}

        // Transfer columns from a Dataframe to a new one 
        Dataframe transfer_col(const std::vector<size_t>& cols_idx);
        Dataframe transfer_col(const std::vector<std::string>& cols_name);
        Dataframe transfer_col(size_t j);  
        Dataframe transfer_col(const std::string& col_name);

        // Delete cols from a Dataframe
        void pop_col(const std::vector<size_t>& cols_idx);
        void pop_col(const std::vector<std::string>& cols_name);
        void pop_col(size_t j);  
        void pop_col(const std::string& col_name);

        // Delete rows from a Dataframe
        void pop_row(const std::vector<size_t>& rows_idx);
        void pop_row(const std::vector<std::string>& rowss_name);
        void pop_row(size_t i);  
        void pop_row(const std::string& row_name);

        // -------------------------Methods change_layout and transpose----------------------------------

        // Change from row - major to col - major inplace, choose between Naive, AVX2...
        Dataframe change_layout(const std::string& choice = "AVX2") const;

        // Change from row - major to col - major inplace, choose between Naive, AVX2...
        void change_layout_inplace(const std::string& choice = "AVX2_threaded");

        // Tranpose Naive
        static std::vector<double> transpose_naive(size_t rows_, size_t cols_, const std::vector<double>& df);

        // Tranpose Naive inplace only for square matrix
        static void transpose_naive_inplace(size_t n, std::vector<double>& df);

        // Transpose Eigen
        static std::vector<double> transpose_eigen(size_t rows_, size_t cols_, const std::vector<double>& df);

        // Transpose Eigen inplace only for square matrix
        static void transpose_eigen_inplace(size_t n, std::vector<double>& df);

        #ifdef __AVX2__
        // Tranpose AVX2 by blocks
        static std::vector<double> transpose_blocks_avx2(size_t rows_, size_t cols_, const std::vector<double>& df);
        
        // Tranpose AVX2 by blocks inplace only for square matrix
        static void transpose_avx2_inplace(size_t n, std::vector<double>& df);

        // Tranpose AVX2_Threaded by blocks
        static std::vector<double> transpose_avx2_th(size_t rows_, size_t cols_, const std::vector<double>& df);
        
        // Tranpose AVX2_Threaded by blocks inplace only for square matrix
        static void transpose_avx2_th_inplace(size_t n, std::vector<double>& df);
        #endif

        #ifdef USE_MKL
        // Tranpose MKL
        static std::vector<double> transpose_mkl(size_t rows_, size_t cols_, const std::vector<double>& df);
        
        // Tranpose inplace MKL only for square matrix
        static void transpose_mkl_inplace(size_t n, std::vector<double>& df);
        #endif
};


class CsvHandler {
    
    public:
        // Returns a column-major Dataframe from Csv path
        static Dataframe loadCsv(const std::string& filepath, 
            char sep = ',', bool is_header = true, const std::string& method = "AVX2_threaded");


    private:
        // Function to encode potential columns using string for categories
        static int encode_label(std::string& label, int col,
            std::unordered_map<int, std::unordered_map<std::string, int>>& label_encoder);
};

