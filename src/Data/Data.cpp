#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"
#include <iomanip>

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

auto Dataframe::asEigen() {
    return Eigen::Map<Eigen::MatrixXd>(data.data(), rows, cols);
}

auto Dataframe::asEigen() const {
    return Eigen::Map<const Eigen::MatrixXd>(data.data(), rows, cols);
}

auto Dataframe::asEigen(std::vector<double>& d, size_t r, size_t c) {
    return Eigen::Map<Eigen::MatrixXd>(d.data(), r, c);
}

auto Dataframe::asEigen(const std::vector<double>& d, size_t r, size_t c) {
    return Eigen::Map<const Eigen::MatrixXd>(d.data(), r, c);
}

std::string Dataframe::decode_label(int value, int col) const {

    for (const auto& [key, val] : label_encoder.at(col)) {
        if (val == value) return key;
    }
    return "NaN - Issue";
}

void Dataframe::display_raw(size_t nb_rows, int space) const {

    std::cout << std::left <<  "Displaying Raw Matrix:" << std::endl;

    // Displaying headers
    for (const auto& s : headers) {
        std::cout << std::setw(space) << s;
    }
    std::cout << std::endl;
    std::cout << std::string(space*headers.size(), '-') << std::endl;

    for (size_t i = 0; i < nb_rows; i++) {
        for (size_t j = 0; j < cols; j++) {

            std::cout << std::setw(space) << (*this)(i,j) ;
        }
        // If end of row
        std::cout << std::endl;
    }
}

void Dataframe::display_decoded(size_t nb_rows, int space) const {

    std::cout << "Displaying Decoded Matrix:" << std::endl;
    
    // Displaying headers
    for (const auto& s : headers) {
        std::cout << std::setw(space) << s;
    }
    std::cout << std::endl;
    std::cout << std::string(space*headers.size(), '-') << std::endl;

    for (size_t i = 0; i < nb_rows; i++) {
        for (size_t j = 0; j < cols; j++) {

            // If current col is encoded then decode value
            if (encoded_cols.find(j) != encoded_cols.end() ) {
                    std::cout << std::setw(space) << decode_label((*this)(i,j), j);
                }
            else std::cout << std::setw(space) << (*this)(i,j);
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

    // Get label_encoder of the col
    std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder_y;
    if (label_encoder.find(static_cast<int>(j)) != label_encoder.end()) {
        label_encoder_y[0] = std::move(label_encoder[static_cast<int>(j)]); 
        label_encoder.erase(static_cast<int>(j));
    }
    
    // Need to fix the indexes of others cols
    std::unordered_map<int, std::unordered_map<std::string, int>> new_label_encoder;
    for (const auto& [col_idx, encoder] : label_encoder) {
        int new_idx = col_idx > static_cast<int>(j) ? col_idx - 1 : col_idx;
        new_label_encoder[new_idx] = encoder;
    }
    label_encoder = std::move(new_label_encoder);

    cols--;

    return {rows, 1, false, std::move(col_y), std::move(headers_y), 
        std::move(label_encoder_y), std::move(encoded_cols_y)};
}

Dataframe Dataframe::transfer_col(const std::string& col_name) {

    // Find col
    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return transfer_col(static_cast<size_t>(idx - headers.begin()));
    else {
        std::cout << "Column not found - try again" << std::endl;
        return {};
    }
}

Dataframe Dataframe::change_layout(const std::string& choice) const {
    
    size_t temp_i, temp_j;
    std::vector<double> new_data;

    if (is_row_major) temp_i = cols, temp_j = rows;
    else temp_i = rows, temp_j = cols;
    
    if (choice == "Naive") new_data = transpose_naive(temp_i, temp_j, data);
    else if (choice == "Eigen") new_data = transpose_eigen(temp_i, temp_j, data);

    #ifdef __AVX2__
        else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
        else new_data = transpose_blocks_avx2(temp_i, temp_j, data);
    #else
        else new_data = transpose_naive(temp_i, temp_j, data);
    #endif

    return {rows, cols, !is_row_major, std::move(new_data), headers, 
        label_encoder, encoded_cols};
}

void Dataframe::change_layout_inplace(const std::string& choice) {
    
    size_t temp_i, temp_j;
    std::vector<double> new_data;

    if (is_row_major) temp_i = cols, temp_j = rows;
    else temp_i = rows, temp_j = cols;

    if (temp_i == temp_j) {
        if (choice == "Naive") transpose_naive_inplace(temp_i, data);
        else if (choice == "Eigen") transpose_eigen_inplace(temp_i, data);

        #ifdef __AVX2__
            else if (choice == "AVX2") transpose_avx2_inplace(temp_i, data);
            else transpose_avx2_inplace(temp_i, data);
        #else
            transpose_naive_inplace(temp_i, data);
        #endif
    }
    else {
        if (choice == "Naive") new_data = transpose_naive(temp_i, temp_j, data);
        else if (choice == "Eigen") new_data = transpose_eigen(temp_i, temp_j, data);
        
        #ifdef __AVX2__
            else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
            else new_data = transpose_blocks_avx2(temp_i, temp_j, data);
        #else
            else new_data = transpose_naive(temp_i, temp_j, data);
        #endif

        data = std::move(new_data);
    }    
    is_row_major = !is_row_major;
}

std::vector<double> Dataframe::transpose_naive(size_t rows_, size_t cols_, 
    const std::vector<double>& df) {
    
    std::vector<double> new_data(rows_*cols_);

    for (size_t i = 0; i < rows_; i++) {
        for(size_t j = 0; j < cols_; j++) {

            new_data[i*cols_ + j] = df[j*rows_ + i];
        }
    }   
    return new_data;
}

void Dataframe::transpose_naive_inplace(size_t n, std::vector<double>& df) {
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            std::swap(df[i*n + j], df[j*n + i]);
        }
    }   
}

std::vector<double> Dataframe::transpose_eigen(size_t rows_, size_t cols_, 
    const std::vector<double>& df) {

    std::vector<double> new_data(rows_*cols_);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), rows_, cols_);
    
    auto eigen_mat = asEigen(df, rows_, cols_);
    res = eigen_mat.transpose();
    
    return new_data;
}

void Dataframe::transpose_eigen_inplace(size_t n, std::vector<double>& df) {
    auto eigen_mat = asEigen(df, n, n);
    eigen_mat.transpose();
}

#ifdef __AVX2__
std::vector<double> Dataframe::transpose_blocks_avx2(size_t rows_, size_t cols_, 
    const std::vector<double>& df) {
    
    std::vector<double> new_data(rows_*cols_);
    
    // Variables
    size_t i = 0, j = 0;
    size_t vec_sizei = rows_ - (rows_ % NB_DB);
    size_t vec_sizej = cols_ - (cols_ % NB_DB);

    for (; i < vec_sizei; i += NB_DB) {
        for (; j < vec_sizej; j += NB_DB) {

            if (j + PREFETCH_DIST1 < vec_sizej) {
                _mm_prefetch((const char*)&df[(j+PREFETCH_DIST1+0) * rows_ + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(j+PREFETCH_DIST1+1) * rows_ + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(j+PREFETCH_DIST1+2) * rows_ + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(j+PREFETCH_DIST1+3) * rows_ + i], _MM_HINT_T0);
            }
            
            // Load 4 cols
            __m256d col0 = _mm256_loadu_pd(&df[(j+0)*rows_ + i]);
            __m256d col1 = _mm256_loadu_pd(&df[(j+1)*rows_ + i]);
            __m256d col2 = _mm256_loadu_pd(&df[(j+2)*rows_ + i]);
            __m256d col3 = _mm256_loadu_pd(&df[(j+3)*rows_ + i]);
            
            // Get pair elements of each
            __m256d t0 = _mm256_unpacklo_pd(col0, col1);

            // Get odd elements of each 
            __m256d t1 = _mm256_unpackhi_pd(col0, col1);

            __m256d t2 = _mm256_unpacklo_pd(col2, col3);
            __m256d t3 = _mm256_unpackhi_pd(col2, col3);
            
            // Get two first elements of each 
            __m256d row0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d row1 = _mm256_permute2f128_pd(t1, t3, 0x20);

            // Get two last elements of each
            __m256d row2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d row3 = _mm256_permute2f128_pd(t1, t3, 0x31);
            
            _mm256_storeu_pd(&new_data[(i+0)*cols_ + j], row0);
            _mm256_storeu_pd(&new_data[(i+1)*cols_ + j], row1);
            _mm256_storeu_pd(&new_data[(i+2)*cols_ + j], row2);
            _mm256_storeu_pd(&new_data[(i+3)*cols_ + j], row3);
        }

        // Scalar residual
        for(; j < cols_; j++) {
            new_data[(i+0)*cols_ + j] = df[j*rows_ + (i+0)];
            new_data[(i+1)*cols_ + j] = df[j*rows_ + (i+1)];
            new_data[(i+2)*cols_ + j] = df[j*rows_ + (i+2)];
            new_data[(i+3)*cols_ + j] = df[j*rows_ + (i+3)];
        }
        j = 0;
    }

    // Scalar residual 
    for (; i < rows_; i++) {
        for(size_t j = 0; j < cols_; j++) {
            new_data[i*cols_ + j] = df[j*rows_ + i];
        }
    }   

    return new_data;
}

void Dataframe::transpose_avx2_inplace(size_t n, std::vector<double>& df) {
    
    const size_t vec_size = n - (n % NB_DB);
    
    // AVX2 blocks (only triangular up)
    for (size_t i = 0; i < vec_size; i += NB_DB) {

        size_t pi = i + PREFETCH_DIST1*NB_DB;
        if (pi < vec_size) {
            _mm_prefetch((const char*)&df[(pi+0) * n + pi], _MM_HINT_T0);
            _mm_prefetch((const char*)&df[(pi+1) * n + pi], _MM_HINT_T0);
            _mm_prefetch((const char*)&df[(pi+2) * n + pi], _MM_HINT_T0);
            _mm_prefetch((const char*)&df[(pi+3) * n + pi], _MM_HINT_T0);
        }
        
        // Diagonal block
        {
            __m256d col0 = _mm256_loadu_pd(&df[(i+0)*n + i]);
            __m256d col1 = _mm256_loadu_pd(&df[(i+1)*n + i]);
            __m256d col2 = _mm256_loadu_pd(&df[(i+2)*n + i]);
            __m256d col3 = _mm256_loadu_pd(&df[(i+3)*n + i]);

            __m256d t0 = _mm256_unpacklo_pd(col0, col1);
            __m256d t1 = _mm256_unpackhi_pd(col0, col1);
            __m256d t2 = _mm256_unpacklo_pd(col2, col3);
            __m256d t3 = _mm256_unpackhi_pd(col2, col3);

            __m256d row0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d row1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d row2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d row3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            _mm256_storeu_pd(&df[(i+0)*n + i], row0);
            _mm256_storeu_pd(&df[(i+1)*n + i], row1);
            _mm256_storeu_pd(&df[(i+2)*n + i], row2);
            _mm256_storeu_pd(&df[(i+3)*n + i], row3);
        }
        
        // Off-diagonal blocks
        for (size_t j = i + NB_DB; j < vec_size; j += NB_DB) {
            
            size_t pj = j + PREFETCH_DIST1*NB_DB;
            if (pj < vec_size) {
                _mm_prefetch((const char*)&df[(i+0)*n + pj], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(i+1)*n + pj], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(i+2)*n + pj], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(i+3)*n + pj], _MM_HINT_T0);
                
                _mm_prefetch((const char*)&df[(pj+0)*n + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(pj+1)*n + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(pj+2)*n + i], _MM_HINT_T0);
                _mm_prefetch((const char*)&df[(pj+3)*n + i], _MM_HINT_T0);
            }
            
            // Load A (i,j)
            __m256d a0 = _mm256_loadu_pd(&df[(i+0)*n + j]);
            __m256d a1 = _mm256_loadu_pd(&df[(i+1)*n + j]);
            __m256d a2 = _mm256_loadu_pd(&df[(i+2)*n + j]);
            __m256d a3 = _mm256_loadu_pd(&df[(i+3)*n + j]);

            // Load B (j,i)
            __m256d b0 = _mm256_loadu_pd(&df[(j+0)*n + i]);
            __m256d b1 = _mm256_loadu_pd(&df[(j+1)*n + i]);
            __m256d b2 = _mm256_loadu_pd(&df[(j+2)*n + i]);
            __m256d b3 = _mm256_loadu_pd(&df[(j+3)*n + i]);

            // Transpose A
            __m256d t0 = _mm256_unpacklo_pd(a0, a1);
            __m256d t1 = _mm256_unpackhi_pd(a0, a1);
            __m256d t2 = _mm256_unpacklo_pd(a2, a3);
            __m256d t3 = _mm256_unpackhi_pd(a2, a3);
            __m256d a_t0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d a_t1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d a_t2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d a_t3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            // Transpose B
            t0 = _mm256_unpacklo_pd(b0, b1);
            t1 = _mm256_unpackhi_pd(b0, b1);
            t2 = _mm256_unpacklo_pd(b2, b3);
            t3 = _mm256_unpackhi_pd(b2, b3);
            __m256d b_t0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d b_t1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d b_t2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d b_t3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            // Swap - B^T, A^T 
            _mm256_storeu_pd(&df[(i+0)*n + j], b_t0);
            _mm256_storeu_pd(&df[(i+1)*n + j], b_t1);
            _mm256_storeu_pd(&df[(i+2)*n + j], b_t2);
            _mm256_storeu_pd(&df[(i+3)*n + j], b_t3);

            _mm256_storeu_pd(&df[(j+0)*n + i], a_t0);
            _mm256_storeu_pd(&df[(j+1)*n + i], a_t1);
            _mm256_storeu_pd(&df[(j+2)*n + i], a_t2);
            _mm256_storeu_pd(&df[(j+3)*n + i], a_t3);
        }
        
        // Scalar residual for j
        for (size_t j = vec_size; j < n; j++) {
            if (i+0 < j) std::swap(df[(i+0)*n + j], df[j*n + (i+0)]);
            if (i+1 < j) std::swap(df[(i+1)*n + j], df[j*n + (i+1)]);
            if (i+2 < j) std::swap(df[(i+2)*n + j], df[j*n + (i+2)]);
            if (i+3 < j) std::swap(df[(i+3)*n + j], df[j*n + (i+3)]);
        }
    }
    
    // Scalar residual for i 
    for (size_t i = vec_size; i < n; i++) {
        for (size_t j = i+1; j < n; j++) {
            std::swap(df[i*n + j], df[j*n + i]);
        }
    }
}
#endif

/*----------------------------------------CsvHandler-----------------------------------*/

int CsvHandler::encode_label(std::string& label, int col, 
    std::unordered_map<int, std::unordered_map<std::string, int>>& label_encoder) {

    // Find if label in vector
    auto it = label_encoder[col].find(label);

    // Encode label
    if (it == label_encoder[col].end()) {
        int new_id;

        // If bool col
        if ((label == "True" || label == "true" || label == "False" || label == "false") && label_encoder[col].size() != 0) {

            if (label == "True" || label == "true") new_id = 1;
            else new_id = 0;

            label_encoder[col]["True"] = 1;
            label_encoder[col]["False"] = 0;
        }
        else if (label == "") {
            new_id = -999999; // NaN
            label_encoder[col][label] = new_id;
        }
        else {
            new_id = label_encoder[col].size();
            label_encoder[col][label] = new_id;
        }
        return new_id;
    }
    return it->second;
}

Dataframe CsvHandler::loadCsv(const std::string& filepath, char sep, bool is_header, const std::string& method) {

    // Class Dataframe variables
    size_t rows = 0, cols = 0;
    std::vector<double> data;
    std::vector<std::string> headers;
    std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder;
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
            if (rows == 0 && is_header) {
                headers.push_back(cell);
            }
            else {
                try {
                    data.push_back(std::stod(cell));

                } catch (const std::invalid_argument&) {
                     
                    // If col of strings
                    int val = encode_label(cell, current_cols, label_encoder); 
                    data.push_back(val);

                    // Get indexes of encoded_cols
                    if (rows == 1) encoded_cols.insert(current_cols); 
                }
            }
            current_cols++;

        }
        if (rows == 1) cols = current_cols;
        rows++;
    }

    if (is_header) rows--;

    Dataframe csv = {rows, cols, true, std::move(data), std::move(headers), 
        std::move(label_encoder), std::move(encoded_cols)};
    
        csv.change_layout_inplace(method);
    // return column-major dataframe
    return csv;
}