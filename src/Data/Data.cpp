#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"

/*----------------------------------------Dataframe-----------------------------------*/

double Dataframe::operator()(size_t i, size_t j) const {
    //assert(i < rows && j < cols);
    return is_row_major ? data[i*cols+j] : data[j*rows+i];
}

const double& Dataframe::at(size_t idx) const {
    return data[idx];
}

double& Dataframe::at(size_t idx) {
    return data[idx];
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

            std::cout << (*this)(i,j) << "  ";
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
            else std::cout << (*this)(i,j) << "  ";
        }
        // If end of row
        std::cout << std::endl;
    }
}

Dataframe Dataframe::transfer_col(size_t j) {

    if (is_row_major) this->change_layout_inplace();  

    // Get Data and erase it
    std::vector<double> col_y(data.begin() + j*rows, data.begin() + (j+1)*rows);
    data.erase(data.begin() + j*rows, data.begin() + (j+1)*rows);
    
    // Get header and erase it
    std::vector<std::string> headers_y = {std::move(headers[j])};
    headers.erase(headers.begin() + j);

    // Get encoded_labels or not
    std::unordered_set<int> encoded_cols_y;
    if (encoded_cols.erase(static_cast<int>(j))) {
        encoded_cols_y.insert(0); 
    }
    
    // Need to fix the indexes of others cols
    std::unordered_set<int> updated_encoded;
    for (int idx : encoded_cols) {
        updated_encoded.insert(idx > static_cast<int>(j) ? idx - 1 : idx);
    }
    encoded_cols = std::move(updated_encoded);

    cols--;

    // By precaution since we have not created a direct link, 
    // we are not sure that certain values belong to our columns
    return {rows, 1, false, std::move(col_y), std::move(headers_y), 
        std::move(label_encoder), std::move(encoded_cols_y)};
}

Dataframe Dataframe::transfer_col(const std::string& col_name) {

    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return transfer_col(static_cast<size_t>(idx - headers.begin()));
    else {
        std::cout << "Column not found - try again" << std::endl;
        return {};
    }
}

Dataframe Dataframe::change_layout() const {
    
    size_t temp_i, temp_j;
    std::vector<double> new_data;
    new_data.reserve(rows * cols);

    if (is_row_major) temp_i = cols, temp_j = rows;
    else temp_i = rows, temp_j = cols;

    for (size_t i = 0; i < temp_i; i++) {
        for(size_t j = 0; j < temp_j; j++) {

            new_data.push_back(data[j*temp_i+i]);
        }
    }
    return {rows, cols, !is_row_major, std::move(new_data), headers, 
        label_encoder, encoded_cols};
}

void Dataframe::change_layout_inplace() {
    
    size_t temp_i, temp_j;
    std::vector<double> new_data;
    new_data.reserve(rows * cols);

    if (is_row_major) temp_i = cols, temp_j = rows;
    else temp_i = rows, temp_j = cols;

    for (size_t i = 0; i < temp_i; i++) {
        for(size_t j = 0; j < temp_j; j++) {

            new_data.push_back(data[j*temp_i+i]);
        }
    }
    is_row_major = !is_row_major;
    data = std::move(new_data);
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

    Dataframe csv = {rows-1, cols, true, std::move(data), std::move(headers)};
    
    // return column-major dataframe
    return csv.change_layout();
}