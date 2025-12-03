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

        // Return corresponding label from a value
        std::string decode_label(int value) const;

        // Displaying our datas either with encoded values or decoded values
        void display_raw(size_t nb_rows) const;
        void display_decoded(size_t nb_rows) const;

        void display_raw() const {display_raw(rows);}
        void display_decoded() const {display_decoded(rows);}

        // Take a column from a Dataframe to create a another Df having this col.
        Dataframe transfer_col(size_t j);  
        Dataframe transfer_col(const std::string& col_name);

        // Change from row - major to col - major
        Dataframe change_layout() const;

        // Change from row - major to col - major 
        void change_layout_inplace();

    // Getters & Constructor
    public:

        // Getting val(i, j) according to our config  
        double operator()(size_t i, size_t j) const;
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
        static Dataframe loadCsv(const std::string& filepath, char sep = ',');

    private:
        // Used to encode potential columns using string for categories
        static int encode_label(std::string& label, std::unordered_map<std::string, int>& label_encoder);
};

