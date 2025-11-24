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
        std::vector<double> data;
        std::vector<std::string> headers;
        std::unordered_map<std::string, int> label_encoder;
        std::unordered_set<int> encoded_cols;

    public: 

        // Get Value
        double operator()(size_t i, size_t j) const;

        // Return corresponding label from a value
        std::string decode_label(int value) const;

        // Displaying our datas either with encoded values or decoded values
        void display_raw(size_t nb_rows) const;
        void display_decoded(size_t nb_rows) const;

        void display_raw() const {display_raw(rows);}
        void display_decoded() const {display_decoded(rows);}

    // Getters & Constructor
    public:

        size_t get_rows() const { return rows; }
        size_t get_cols() const { return cols; }

        Dataframe(size_t r, size_t c, std::vector<double> d, std::vector<std::string> h,
            std::unordered_map<std::string, int> l, std::unordered_set<int> e)
            : rows(r), cols(c), data(std::move(d)), headers(std::move(h)), 
            label_encoder(std::move(l)), encoded_cols(std::move(e)) {}

};

class CsvHandler {
    
    public:
        // Returns Dataframe from Csv path
        static Dataframe loadCsv(const std::string& filepath, char sep = ',');

    private:
        // Used to encode potential columns using string for categories
        static int encode_label(std::string& label, std::unordered_map<std::string, int>& label_encoder);
};

