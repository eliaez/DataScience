#include <iomanip>
#include <Eigen/Dense>
#include <system_error>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Linalg/Linalg.hpp"
#include "Utils/ThreadPool.hpp"

#ifdef USE_MKL
    #include <mkl.h>
#endif

using namespace Utils;

/*----------------------------------------Dataframe-----------------------------------*/
// ---------------------------------Operators or Equivalents---------------------------

const double& Dataframe::at(size_t idx) const {
    return data[idx];
}

double& Dataframe::at(size_t idx) {
    return data[idx];
}

const double& Dataframe::operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("i or j > of your Dataframe dimensions");
    }
    return is_row_major ? data[i*cols+j] : data[j*rows+i];
}

double& Dataframe::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("i or j > of your Dataframe dimensions");
    }
    return is_row_major ? data[i*cols+j] : data[j*rows+i];
}

Dataframe Dataframe::operator[](const std::vector<size_t>& cols_idx) const {

    // Test
    for (const size_t& idx : cols_idx) {
        if (idx >= cols) {
            throw std::out_of_range("One of yours indexes is > of Dataframe dimensions");
        }    
    }

    // Output
    size_t nb = cols_idx.size();
    std::vector<double> y_data;
    y_data.reserve(rows * cols_idx.size());
    std::vector<std::string> headers_y;
    std::unordered_set<int> encoded_cols_y;
    std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder_y;

    // Sort our vector
    std::vector<size_t> idx_sorted = cols_idx;
    std::sort(idx_sorted.begin(), idx_sorted.end());

    // Extract data
    int i = 0;
    if (is_row_major) {
        for (const size_t& idx : cols_idx) {

            // Get data
            for (size_t j = 0; j < rows; j++) {
                y_data.push_back(data[j*cols+idx]);
            }

            // Get header
            headers_y.push_back(headers[idx]);

            // Get encoded cols idx
            if (encoded_cols.find(static_cast<int>(idx)) != encoded_cols.end()) encoded_cols_y.insert(i);

            // Get mapping
            if (label_encoder.contains(static_cast<int>(idx))) label_encoder_y.insert({i, label_encoder.at(static_cast<int>(idx))});
            i++;
        }
    }
    else {
        for (const size_t& idx : cols_idx) {

            // Get data
            for (size_t j = 0; j < rows; j++) {
                y_data.push_back(data[idx*rows+j]);
            }

            // Get header
            headers_y.push_back(headers[idx]);

            // Get encoded cols idx
            if (encoded_cols.find(static_cast<int>(idx)) != encoded_cols.end()) encoded_cols_y.insert(i);

            // Get mapping
            if (label_encoder.contains(static_cast<int>(idx))) label_encoder_y.insert({i, label_encoder.at(static_cast<int>(idx))});
            i++;
        }
    }
    return {rows, nb, false, std::move(y_data), std::move(headers_y), 
        std::move(label_encoder_y), std::move(encoded_cols_y)};
}

Dataframe Dataframe::operator[](const std::vector<std::string>& cols_name) const {
    
    // Find cols idx
    std::vector<size_t> cols_idx;
    cols_idx.reserve(cols_name.size());
    for (const std::string& name : cols_name) {

        auto idx = std::find(headers.begin(), headers.end(), name);
        if (idx != headers.end()) cols_idx.push_back(static_cast<size_t>(idx - headers.begin()));
        else throw std::invalid_argument(std::format("Column {} not found", name));
    }
    return operator[](cols_idx);
}

Dataframe Dataframe::operator[](std::initializer_list<int> cols_idx) const {
    return operator[](std::vector<size_t>(cols_idx.begin(), cols_idx.end()));
}

Dataframe Dataframe::operator[](std::initializer_list<std::string> cols_name) const {
    return operator[](std::vector<std::string>(cols_name));
}

Dataframe Dataframe::operator[](size_t j) const {
    std::vector<size_t> cols_idx = {j};
    return operator[](cols_idx);
}

Dataframe Dataframe::operator[](const std::string& col_name) const {
    std::vector<std::string> cols_name = {col_name};
    return operator[](cols_name);
}

Dataframe Dataframe::operator+(const Dataframe& other) const { return Linalg::Operations::sum(*this, other); }
Dataframe Dataframe::operator-(const Dataframe& other) const { return Linalg::Operations::sum(*this, other, '-'); }
Dataframe Dataframe::operator*(const Dataframe& other) const { return Linalg::Operations::multiply(*this, other); }
Dataframe Dataframe::operator~() { return Linalg::Operations::transpose(*this);}
Dataframe Dataframe::inv() { return Linalg::Operations::inverse(*this); }
std::tuple<double, std::vector<double>, std::vector<double>> Dataframe::det() { return Linalg::Operations::determinant(*this); }
int Dataframe::is_tri() const { return Linalg::Operations::triangular_matrix(*this); }

// ---------------------------------------Methods---------------------------------------

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

void Dataframe::OneHot(size_t j) {
    
    if (is_row_major) this->change_layout_inplace();

    // Get label_encoder of the col which is by default encoded
    std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder_y;
    if (label_encoder.find(static_cast<int>(j)) != label_encoder.end()) {
        label_encoder_y[0] = label_encoder[static_cast<int>(j)]; 
    }
    else {
        throw std::invalid_argument(std::format("Column {} not found in label encoder", j));
    }

    // Get data of the col 
    std::vector<double> to_convert = this->popup(j);

    // Create a new col for each category and data for each rows
    size_t idx = 0;
    std::vector<double> onehot_data(rows * label_encoder_y[0].size());
    for (auto& [str, val] : label_encoder_y[0]) {

        // To avoid NAN col
        if (str != "") {
            headers.push_back(str);
            for (size_t k = 0; k < rows; k++) {
                onehot_data[rows * idx + k] = (to_convert[k] == val) ? 1.0 : 0.0;
            }
            idx++;
            cols++;
        }
    }
    onehot_data.shrink_to_fit();

    // Insert our new data at the end
    data.reserve(data.size() + onehot_data.size());
    data.insert(data.end(), onehot_data.begin(), onehot_data.end());
}

void Dataframe::OneHot(const std::string& col_name) {

    // Find col
    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return OneHot(static_cast<size_t>(idx - headers.begin()));
    else {
        throw std::invalid_argument(std::format("Column {} not found", col_name));
    }
}

Dataframe Dataframe::transfer_col(size_t j) {

    if (is_row_major) this->change_layout_inplace();  

    // Get Data and erase it
    std::vector<double> col_y(data.begin() + j*rows, data.begin() + (j+1)*rows);
    data.erase(data.begin() + j*rows, data.begin() + (j+1)*rows);
    
    // Get header and erase it
    std::vector<std::string> headers_y;
    if (!headers.empty() && j < headers.size()) {
        headers_y = {std::move(headers[j])};
        headers.erase(headers.begin() + j);
    }

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
        throw std::invalid_argument(std::format("Column {} not found", col_name));
    }
}

Dataframe Dataframe::transfer_col(const std::vector<size_t>& cols_idx) {

    // Output
    std::vector<double> col_y; 
    std::vector<std::string> headers_y;
    std::unordered_set<int> encoded_cols_y;
    std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder_y;

    // Sort our vector
    std::vector<size_t> idx_sorted = cols_idx;
    std::sort(idx_sorted.begin(), idx_sorted.end());

    // Get data for each idx
    int i = 0;
    Dataframe df_idx;
    for (const size_t& idx : idx_sorted){
        df_idx = transfer_col(idx-i);

        // Update our output
        col_y.insert(col_y.end(), df_idx.get_data().begin(), df_idx.get_data().end());
        headers_y.insert(headers_y.end(), df_idx.get_headers().begin(), df_idx.get_headers().end());


        if (!df_idx.get_encodedCols().empty()) encoded_cols_y.insert(i);

        auto lab_enc = df_idx.get_encoder();
        if (!lab_enc.empty()) label_encoder_y.insert({i, lab_enc[0]});

        i++;
    }

    return {rows, cols_idx.size(), false, std::move(col_y), std::move(headers_y), 
        std::move(label_encoder_y), std::move(encoded_cols_y)};
}

Dataframe Dataframe::transfer_col(const std::vector<std::string>& cols_name) {

    // Find cols idx
    std::vector<size_t> cols_idx;
    cols_idx.reserve(cols_name.size());
    for (const std::string& name : cols_name) {

        auto idx = std::find(headers.begin(), headers.end(), name);
        if (idx != headers.end()) cols_idx.push_back(static_cast<size_t>(idx - headers.begin()));
        else throw std::invalid_argument(std::format("Column {} not found", name));
    }
    return transfer_col(cols_idx);
}

Dataframe Dataframe::transfer_col(std::initializer_list<int> cols_idx) {
    return transfer_col(std::vector<size_t>(cols_idx.begin(), cols_idx.end()));
}

Dataframe Dataframe::transfer_col(std::initializer_list<std::string> cols_name) {
    return transfer_col(std::vector<std::string>(cols_name));
}

std::vector<double> Dataframe::popup(const std::vector<size_t>& v_idx, bool is_row) {

    // Delete and return rows
    if (is_row) {
        if (is_row_major) this->change_layout_inplace();

        // Sort our vector
        size_t nb = v_idx.size();
        std::vector<size_t> idx_sorted = v_idx;
        std::sort(idx_sorted.begin(), idx_sorted.end());

        // To know which rows will be kept
        std::vector<bool> keep(rows, true);
        for (size_t idx : idx_sorted) keep[idx] = false;
        
        std::vector<double> res;
        res.reserve(cols * nb);
        std::vector<double> new_data;
        new_data.reserve(cols * (rows - nb));
        
        for (size_t c = 0; c < cols; c++) {
            size_t offset = c * rows;
            for (size_t r = 0; r < rows; r++) {
                if (!keep[r]) {
                    res.push_back(data[offset + r]);
                } else {
                    new_data.push_back(data[offset + r]);
                }
            }
        }

        rows -= nb;
        data = std::move(new_data);
        return res;
    }
    // Delete and return cols
    else {
        return transfer_col(v_idx).get_data();
    }
}

std::vector<double> Dataframe::popup(const std::vector<std::string>& cols_name) {
    // Find cols idx
    std::vector<size_t> cols_idx;
    cols_idx.reserve(cols_name.size());
    for (const std::string& name : cols_name) {

        auto idx = std::find(headers.begin(), headers.end(), name);
        if (idx != headers.end()) cols_idx.push_back(static_cast<size_t>(idx - headers.begin()));
        else throw std::invalid_argument(std::format("Column {} not found", name));
    }
    return popup(cols_idx);
}

std::vector<double> Dataframe::popup(std::initializer_list<int> v_idx, bool is_row) {
    return popup(std::vector<size_t>(v_idx.begin(), v_idx.end()), is_row);
}

std::vector<double> Dataframe::popup(std::initializer_list<std::string> cols_name) {
    return popup(std::vector<std::string>(cols_name));
}

std::vector<double> Dataframe::popup(size_t j, bool is_row) {
    std::vector<size_t> v_idx = {j};
    return popup(v_idx, is_row);
}

std::vector<double> Dataframe::popup(const std::string& col_name) {
    std::vector<std::string> cols_name = {col_name};
    return popup(cols_name);
}

void Dataframe::pop(const std::vector<size_t>& v_idx, bool is_row) {
    auto res = popup(v_idx, is_row);
}

void Dataframe::pop(const std::vector<std::string>& cols_name) {
    auto res = popup(cols_name);
}

void Dataframe::pop(std::initializer_list<int> v_idx, bool is_row) {
    auto res = popup(std::vector<size_t>(v_idx.begin(), v_idx.end()), is_row);
}

void Dataframe::pop(std::initializer_list<std::string> cols_name) {
    auto res = popup(std::vector<std::string>(cols_name));
}

void Dataframe::pop(size_t j, bool is_row) {
    auto res = popup(j, is_row);
}

void Dataframe::pop(const std::string& col_name) {
    auto res = popup(col_name);
}
    
// -------------------------Methods change_layout and transpose----------------------------------

Dataframe Dataframe::change_layout(const std::string& choice) const {
    
    size_t temp_i, temp_j;
    std::vector<double> new_data;

    if (is_row_major) temp_i = cols, temp_j = rows;
    else temp_i = rows, temp_j = cols;
    
    if (choice == "Naive") new_data = transpose_naive(temp_i, temp_j, data);
    else if (choice == "Eigen") new_data = transpose_eigen(temp_i, temp_j, data);

    #if defined(__AVX2__) && defined(USE_MKL)
        else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
        else if (choice == "AVX2_threaded") new_data = transpose_avx2_th(temp_i, temp_j, data);
        else if (choice == "MKL") new_data = transpose_mkl(temp_i, temp_j, data);
        else new_data = transpose_avx2_th(temp_i, temp_j, data);
    #elif defined(__AVX2__)
        else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
        else if (choice == "AVX2_threaded") new_data = transpose_avx2_th(temp_i, temp_j, data);
        else new_data = transpose_avx2_th(temp_i, temp_j, data);
    #elif defined(USE_MKL)
        else if (choice == "MKL") new_data = transpose_mkl(temp_i, temp_j, data);
        else new_data = transpose_naive(temp_i, temp_j, data);
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

        #if defined(__AVX2__) && defined(USE_MKL)
            else if (choice == "AVX2") transpose_avx2_inplace(temp_i, data);
            else if (choice == "AVX2_threaded") transpose_avx2_th_inplace(temp_i, data);
            else if (choice == "MKL") transpose_mkl_inplace(temp_i, data);
            else transpose_avx2_th_inplace(temp_i, data);
        #elif defined(__AVX2__)
            else if (choice == "AVX2") transpose_avx2_inplace(temp_i, data);
            else if (choice == "AVX2_threaded") transpose_avx2_th_inplace(temp_i, data);
            else transpose_avx2_th_inplace(temp_i, data);
        #elif defined(USE_MKL)
            else if (choice == "MKL") transpose_mkl_inplace(temp_i, data);
            else transpose_naive_inplace(temp_i, data);
        #else
            transpose_naive_inplace(temp_i, data);
        #endif
    }
    else {
        if (choice == "Naive") new_data = transpose_naive(temp_i, temp_j, data);
        else if (choice == "Eigen") new_data = transpose_eigen(temp_i, temp_j, data);
        
        #if defined(__AVX2__) && defined(USE_MKL)
            else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
            else if (choice == "AVX2_threaded") new_data = transpose_avx2_th(temp_i, temp_j, data);
            else if (choice == "MKL") new_data = transpose_mkl(temp_i, temp_j, data);
            else new_data = transpose_avx2_th(temp_i, temp_j, data);
        #elif defined(__AVX2__)
            else if (choice == "AVX2") new_data = transpose_blocks_avx2(temp_i, temp_j, data);
            else if (choice == "AVX2_threaded") new_data = transpose_avx2_th(temp_i, temp_j, data);
            else new_data = transpose_avx2_th(temp_i, temp_j, data);
        #elif defined(USE_MKL)
            else if (choice == "MKL") new_data = transpose_mkl(temp_i, temp_j, data);
            else new_data = transpose_naive(temp_i, temp_j, data);
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
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), cols_, rows_);
    
    auto eigen_mat = Eigen::Map<const Eigen::MatrixXd>(df.data(), rows_, cols_);
    res = eigen_mat.transpose();
    
    return new_data;
}

void Dataframe::transpose_eigen_inplace(size_t n, std::vector<double>& df) {
    auto eigen_mat = Eigen::Map<Eigen::MatrixXd>(df.data(), n, n);
    eigen_mat.transposeInPlace();
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
        for(j = 0; j < cols_; j++) {
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

std::vector<double> Dataframe::transpose_avx2_th(size_t rows_, size_t cols_, 
    const std::vector<double>& df) {

    // After multiple tests a minimum was decided
    constexpr size_t THREADING_THRESHOLD = 512;
    if (rows_ < THREADING_THRESHOLD) {
        return transpose_blocks_avx2(rows_, cols_, df);
    }
    
    // Variables
    std::vector<double> new_data(rows_*cols_);
    size_t vec_sizei = rows_ - (rows_ % NB_DB);
    size_t vec_sizej = cols_ - (cols_ % NB_DB);
    
    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_threads = pool.nb_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(nb_threads);

    size_t chunk = (int)(vec_sizei / nb_threads);
    size_t start_i = 0, end_i = chunk;
    
    for (size_t n = 0; n < nb_threads; n++) {
        if (n+1 == nb_threads) end_i = vec_sizei;

        auto fut = pool.enqueue([start_i, end_i, rows_, cols_, vec_sizej, &df, &new_data] {
            for (size_t i = start_i; i < end_i; i += NB_DB) {
                for (size_t j = 0; j < vec_sizej; j += NB_DB) {

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
                for(size_t j = vec_sizej; j < cols_; j++) {
                    new_data[(i+0)*cols_ + j] = df[j*rows_ + (i+0)];
                    new_data[(i+1)*cols_ + j] = df[j*rows_ + (i+1)];
                    new_data[(i+2)*cols_ + j] = df[j*rows_ + (i+2)];
                    new_data[(i+3)*cols_ + j] = df[j*rows_ + (i+3)];
                }
            }
        });

        futures.push_back(std::move(fut));
        start_i += chunk;
        end_i += chunk;
    }

    for (auto& fut : futures) {
        fut.wait();
    }

    // Scalar residual 
    for (size_t i = vec_sizei; i < rows_; i++) {
        for(size_t j = 0; j < cols_; j++) {
            new_data[i*cols_ + j] = df[j*rows_ + i];
        }
    }   

    return new_data;
}

void Dataframe::transpose_avx2_th_inplace(size_t n, std::vector<double>& df) {
    
    // After multiple tests a minimum was decided
    constexpr size_t THREADING_THRESHOLD = 512;
    if (n < THREADING_THRESHOLD) {
        return transpose_avx2_inplace(n, df);
    }

    size_t vec_size = n - (n % NB_DB);

    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_blocks = vec_size / NB_DB;

    std::vector<std::future<void>> futures;
    futures.reserve(nb_blocks);

    // By diagonals only, to avoid collision
    for (size_t diag = 0; diag < nb_blocks; diag++) {
        auto fut = pool.enqueue([n, diag, &df] {
            
            // Blocks on this diagonal are independant
            for (size_t k = 0; k <= diag; k++) {
                size_t i = k * NB_DB;
                size_t j = (diag - k) * NB_DB;
                
                if (i == j) { // Diagonal Block
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
                    
                } else if (i < j) { // Off-diagonal blocks
                    __m256d a0 = _mm256_loadu_pd(&df[(i+0)*n + j]);
                    __m256d a1 = _mm256_loadu_pd(&df[(i+1)*n + j]);
                    __m256d a2 = _mm256_loadu_pd(&df[(i+2)*n + j]);
                    __m256d a3 = _mm256_loadu_pd(&df[(i+3)*n + j]);

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
            }
        });
        futures.push_back(std::move(fut));
    }

    for (auto& f : futures) f.wait();

    // Scalar résidual for i
    for (size_t i = vec_size; i < n; i++) {
        for (size_t j = i+1; j < n; j++) {
            std::swap(df[i*n + j], df[j*n + i]);
        }
    }
    
    // Scalar résidual for j
    for (size_t i = 0; i < vec_size; i++) {
        for (size_t j = vec_size; j < n; j++) {
            if (i < j) std::swap(df[i*n + j], df[j*n + i]);
        }
    }
}
#endif

#ifdef USE_MKL
std::vector<double> Dataframe::transpose_mkl(size_t rows_, size_t cols_, 
    const std::vector<double>& df) {
    
    std::vector<double> new_data(rows_*cols_);
    mkl_domatcopy(
        'C',              // 'C' for Col-major, else 'R'
        'T',              // 'N' no, 'T' transpose, 'C transpose conjugate
        rows_,
        cols_,          
        1.0,              // Scalar
        df.data(),        // Input
        rows_,            // Leading dim input
        new_data.data(),  // Output
        cols_             // Leading dim output
    );
    return new_data;
}

void Dataframe::transpose_mkl_inplace(size_t n, std::vector<double>& df) {
    mkl_dimatcopy(
        'C',              // 'C' for Col-major, else 'R'
        'T',              // 'N' no, 'T' transpose, 'C' transpose conjugate
        n,
        n,          
        1.0,              // Scalar
        df.data(),        // Input
        n,
        n
    );
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

Dataframe CsvHandler::loadCsv(const std::string& filepath, char delimiter, bool has_header, const std::string& transpose_method) {

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
            " (" + std::generic_category().message(errno) + ")"
        );
    }

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        size_t current_cols = 0;

        while (std::getline(ss, cell, delimiter)) {

            // For the header
            if (rows == 0 && has_header) {
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

    if (has_header) rows--;

    Dataframe csv = {rows, cols, true, std::move(data), std::move(headers), 
        std::move(label_encoder), std::move(encoded_cols)};
    
    // return column-major dataframe
    csv.change_layout_inplace(transpose_method);
    return csv;
}