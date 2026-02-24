# I - Dataframe
As a first step, we need a class to handle and manipulate our data. Thus, our [**Dataframe**](include/Data/Data.hpp) class provides the structure for numerical data manipulation and is specifically designed for machine learning and linear algebra operations. It supports both row-major and column-major layouts by storing data as a contiguous `std::vector<double>`. Additionally, the class handles heterogeneous data types through automatic categorical encoding.

## Core Features

### Internal Structure 

- #### Data Model
  The Dataframe uses a flat memory model optimized for vectorized operations and cache performance.

  ```cpp
  class Dataframe {
      size_t rows, cols;
      bool is_row_major;                                                            // Layout flag
      std::vector<double> data;                                                     // Contiguous storage
      std::vector<std::string> headers;                                             // Column names
      std::unordered_map<int, std::unordered_map<std::string, int>> label_encoder;  // Category mappings
      std::unordered_set<int> encoded_cols;                                         // Encoded column indices
  };
  ```

- #### Memory Layout Management
  The Dataframe implements dual memory layouts: row-major and column-major, with the latter favored for linear algebra due to better cache locality. Moreover, multiple transpose implementations enable efficient layout conversion, aligned with our backend performance study :

    - `Naive`: Basic double-loop transposition
    - `AVX2`: SIMD vectorized 4x4 block transposition with cache prefetching
    - `AVX2_threaded`: Parallel AVX2 using thread pool
    - `Eigen`: Leverages Eigen library optimizations (for comparison purpose)
    - `MKL`: Intel Math Kernel Library (for comparison purpose) <br> <br>

  ```cpp 
  // Change layout by copy with Naive backend (with df col-major)
  df_row_major = df.change_layout("Naive");

  // Change layout in-place with default backend
  df.change_layout_inplace();
  ```

### CSV Loading Pipeline
- #### Parsing and Encoding
  The pipeline parses CSV files with configurable delimiters, detects and encodes categorical columns by converting string values to unique integers in `label_encoder`. Data is then converted to column-major (if initially row-major) using the specified transpose method (default: `AVX2_threaded).

  ```cpp
  Dataframe CsvHandler::loadCsv(
      const std::string& filepath,
      char delimiter = ',',
      bool has_header = true,
      const std::string& transpose_method = "AVX2_threaded"
  );

  auto df = CsvHandler::loadCsv("../data.csv", ',', true, "AVX2_threaded"); // df will be col-major
  ```

- #### Label Encoding
  Automatic categorical encoding handles mixed-type datasets during CSV loading. The `encode_label()` function converts string categories to unique integers, while `decode_label()` enables reversible reconstruction. Furthermore, metadata is preserved through `encoded_cols` (with column indices) and `label_encoder` (with the mapping between string and integer) to ensure encoded information are preserved through transformations. 
  
  Once encoding metadata is no longer needed, `clear_encoding()` frees label_encoder and encoded_cols to reduce memory footprint while preserving data and headers.

  ```cpp 
  // Display your df with encoded columns
  df.display_raw(); 
  df.display_raw(10); // Display 10 first rows 

  // Display your df with decoded columns
  df.display_decoded();
  df.display_decoded(10);

  // Release encoding memory (keeps data and headers)
  df.clear_encoding();
  ```

### Data Manipulation

- #### Access Patterns
  Three access patterns accommodate different use cases:

  - `operator()(i, j)`: provides an indexed access and abstracts layout complexity
  - `at(idx)`: provides direct access for raw vector manipulation
  - `operator[]`: Column extraction returning `Dataframe` <br> <br>

  ```cpp 
  // Get i, j value (be careful of your layout and your indices)
  auto val = df(i,j);
  auto val = df.at(i*n + j);

  // Column extraction
  auto df_col = df[0];                           // By index
  auto df_col = df["target"];                    // By name
  auto df_cols = df[{0, 2, 4}];                  // Multiple indices
  auto df_cols = df[{"feature1", "feature2"}];   // Multiple names
  ```

- #### One-Hot Encoding
  Each unique value in the column will become a new binary column and the original column is removed:

  ```cpp
  auto df_encoded = df.OneHot(2);               // By indices
  auto df_encoded = df.OneHot("HomePlanet");    // By names
  ```

- #### Column Manipulation
  Three operation types for column management:

  - `transfer_col()`: Move columns to new Dataframe with metadata
  - `popup()`: Extract and remove rows/cols, returns data as vector
  - `pop()`: Remove rows/cols without returning data

  All operations support indices, names, vectors, and initializer lists. `transfer_col()`, `popup()`, and `pop()` automatically convert to column-major layout.

  ```cpp
  // Transfer column (moves data)
  auto y = df.transfer_col("target");
  auto features = df.transfer_col({"feature1", "feature2"});

  // Remove and return
  auto last_row = df.popup(df.rows() - 1, true);  // is_row=true
  auto col_data = df.popup({"temp_column"});

  // Remove without return
  df.pop({0, 1, 2}, true);  // Remove first 3 rows
  df.pop("temp1_column");
  ```

- #### Operator Overloading
  The Dataframe interface provides intuitive operators for common operations:

  - Matrix operations (**`+`, `-`, `*`, `~`**)
  - Linear algebra methods (**`inv()`, `det()`, `is_tri()`**)

  ```cpp
  // Matrix operations
  auto sum = df1 + df2;                 // Addition
  auto diff = df1 - df2;                // Subtraction
  auto prod = df1 * df2;                // Multiplication
  auto transp = ~df;                    // Transpose

  // Linear algebra
  auto inverse = df.inv();              // Matrix inverse
  auto [det, Pivots, U] = df.det();     // Determinant with Pivots and LU matrix
  int tri_type = df.is_tri();           // Check triangular (0/1/2/3) ~ (No/Down/Up/Diag)
<br>

**To test it yourself**, you can also check the corresponding files: 
- [**Data.hpp**](/include/Data/Data.hpp)
- [**Data.cpp**](/src/Data/Data.cpp)
- [**Test file**](/tests/tests_data.cpp)

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**II - Linear Algebra**](/docs/II_linalg.md).