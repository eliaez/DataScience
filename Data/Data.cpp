#include "Data.hpp"
#include "Maths/Linalg.hpp"

/*----------------------------------------Dataframe-----------------------------------*/

double Dataframe::operator()(size_t i, size_t j) const {
    //assert(i < rows && j < cols);

    if (is_row_major) return data[i * cols + j];
    else return data[j * rows + i];
}

std::string Dataframe::decode_label(int value) const {

    for (const auto& [key, val] : label_encoder) {
        if (val == value) return key;
    }

    return "NaN - Issue";
}

void Dataframe::display_raw(size_t nb_rows) const {

    std::cout << "Displaying Raw Matrix:" << std::endl;

    // Displaying headers
    for (const auto& s : headers) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < nb_rows; i++) {
        for (size_t j = 0; j < cols; j++) {

            std::cout << std::to_string((*this)(i,j)) << "  ";
        }

        // If end of row
        std::cout << std::endl;
    }
}

void Dataframe::display_decoded(size_t nb_rows) const {

    std::cout << "Displaying Decoded Matrix:" << std::endl;
    
    // Displaying headers
    for (const auto& s : headers) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < nb_rows; i++) {
        for (size_t j = 0; j < cols; j++) {

            // If current col is encoded then decode value
            if (encoded_cols.find(j) != encoded_cols.end() ) {
                    std::cout << decode_label((*this)(i,j)) << "  ";
                }
            else std::cout << std::to_string((*this)(i,j)) << "  ";
        }
        // If end of row
        std::cout << std::endl;
    }
}

/*----------------------------------------CsvHandler-----------------------------------*/

int CsvHandler::encode_label(std::string& label, std::unordered_map<std::string, int>& label_encoder) {

    // Return pointer to the value or end()
    auto it = label_encoder.find(label);

    if (it == label_encoder.end()) {
        int new_id = label_encoder.size();
        label_encoder[label] = new_id;
        return new_id;
    }
    return it->second;
}

Dataframe CsvHandler::loadCsv(const std::string& filepath, char sep) {

    // Class Dataframe variables
    size_t rows = 0, cols = 0;
    std::vector<double> data;
    std::vector<std::string> headers;
    std::unordered_map<std::string, int> label_encoder;
    std::unordered_set<int> encoded_cols;

    // Read file
    std::ifstream file(filepath);
    std::string line;

    if (!file) {
        throw std::runtime_error(
            "Cannot open: " + filepath + 
            " (" + std::strerror(errno) + ")"
        );
    }

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        size_t current_cols = 0;

        while (std::getline(ss, cell, sep)) {

            // For the header
            if (rows == 0) {
                headers.push_back(cell);
            }
            else {
                try {
                    data.push_back(std::stod(cell));

                } catch (const std::invalid_argument&) {
                     
                    // If col of strings
                    int val = encode_label(cell, label_encoder); 
                    data.push_back(val);

                    // Get indexes of encoded_cols
                    if (rows == 1) encoded_cols.insert(current_cols); 
                }
            }
            if (rows == 1) current_cols++;

        }
        if (rows == 1) cols = current_cols;
        rows++;
    }

    // return column-major dataframe
    return Linalg::transpose_naive({rows-1, cols, true, std::move(data), std::move(headers)});
}