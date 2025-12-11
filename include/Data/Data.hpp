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

#ifdef __AVX2__
#include <immintrin.h>
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
        std::unordered_map<std::string, int> label_encoder;
        std::unordered_set<int> encoded_cols;

    public:
        #ifdef __AVX2__
        static constexpr size_t PREFETCH_DIST1 = 16; // Pre-fetch 16*8 bytes ahead for Blocks algo
        #endif

    public: 

        // Return corresponding label from a value
        std::string decode_label(int value) const;

        // Displaying our datas with encoded values
        void display_raw(size_t nb_rows) const;
        void display_raw() const {display_raw(rows);}

        // Displaying our datas with decoded values
        void display_decoded(size_t nb_rows) const;
        void display_decoded() const {display_decoded(rows);}

        // Transfer a column from a Dataframe to a new one 
        Dataframe transfer_col(size_t j);  
        Dataframe transfer_col(const std::string& col_name);

        // Change from row - major to col - major inplace, choose between Naive, AVX2...
        Dataframe change_layout(const std::string& choice = "AVX2") const;

        // Change from row - major to col - major inplace, choose between Naive, AVX2...
        void change_layout_inplace(const std::string& choice = "AVX2");

        // Tranpose Naive
        static std::vector<double> transpose_naive(size_t rows_, size_t cols_, const std::vector<double>& df);

        #ifdef __AVX2__
        // Tranpose AVX2 by blocks (see LinalgAVX2.hpp for NB_DB)
        static std::vector<double> transpose_blocks_avx2(size_t rows_, size_t cols_, const std::vector<double>& df, size_t NB_DB);
        #endif

    // Getters & Constructor
    public:

        // Getting val(i, j) according to our storage config  
        double operator()(size_t i, size_t j) const;

        // Getting value from Dataframe according to index
        const double& at(size_t idx) const;
        double& at(size_t idx);
        
        /*std::vector<double>& row(size_t i); // Getting row i
        std::vector<double>& col(size_t j); // Getting column j
        std::vector<double>& col(const std::string& header); // Getting column with header*/

        size_t get_rows() const { return rows; }
        size_t get_cols() const { return cols; }

        size_t size() const { return data.size(); }
        
        bool get_storage() const {return is_row_major; }

        const std::vector<double>& get_data() const { return data; }
        const std::vector<std::string>& get_headers() const { return headers; }
        const std::unordered_map<std::string, int>& get_encoder() const { return label_encoder; }
        const std::unordered_set<int>& get_encodedCols() const { return encoded_cols; }

        Dataframe(size_t r = 0, size_t c = 0, bool i = true, std::vector<double> d = {}, 
            std::vector<std::string> h = {}) : rows(r), cols(c), is_row_major(i), 
            data(std::move(d)), headers(std::move(h)) {}

        Dataframe(size_t r, size_t c, bool i, std::vector<double> d, std::vector<std::string> h,
            std::unordered_map<std::string, int> l, std::unordered_set<int> e)
            : rows(r), cols(c), is_row_major(i), data(std::move(d)), headers(std::move(h)), 
            label_encoder(std::move(l)), encoded_cols(std::move(e)) {}

};

class CsvHandler {
    
    public:
        // Returns a column-major Dataframe from Csv path
        static Dataframe loadCsv(const std::string& filepath, char sep = ',', bool is_header = true);

    private:
        // Function to encode potential columns using string for categories
        static int encode_label(std::string& label, std::unordered_map<std::string, int>& label_encoder);
};

