# I - Dataframe
As a first step, we need a class to handle and manipulate our data. Thus, our [**Dataframe**](include/Data/Data.hpp) class provides the structure for numerical data manipulation and is specifically designed for machine learning and linear algebra operations. It supports both row-major and column-major layouts by storing data as a contiguous `std::vector<double>`. Additionally, the class handles heterogeneous data types through automatic categorical encoding.

## Core Features

#### Memory Layout Management
The Dataframe implements dual memory layouts: row-major and column-major, with the latter favored for linear algebra due to better cache locality. Moreover, multiple transpose implementations enable efficient layout conversion, aligned with our backend performance study :

  - `Naive`: Basic double-loop transposition
  - `AVX2`: SIMD vectorized 4x4 block transposition with cache prefetching
  - `AVX2_threaded`: Parallel AVX2 using thread pool
  - `Eigen`: Leverages Eigen library optimizations (for comparison purpose)
  - `MKL`: Intel Math Kernel Library (for comparison purpose)

#### CSV Loading (`CsvHandler::loadCsv`)
The pipeline parses CSV files with configurable delimiters, detects and encodes categorical columns by converting string values to unique integers in `label_encoder`. Data is then converted to column-major (if initially row-major) using the specified transpose method (default: `AVX2_threaded`).

#### Label Encoding
Automatic categorical encoding handles mixed-type datasets during CSV loading. The `encode_label()` function converts string categories to unique integers, while `decode_label()` enables reversible reconstruction. Furthermore, metadata is preserved through `encoded_cols` (with column indices) and `label_encoder` (with the mapping between string and integer) to ensure encoded information are preserved through transformations.

#### Data Access
Three access patterns accommodate different use cases:

- `operator()(i, j)`: provides an indexed access and abstracts layout complexity
- `at(idx)`: provides direct access for raw vector manipulation
- `asEigen()`: provides zero-copy Eigen matrix views (to enable the Eigen backend use case)

#### Column Operations
The `transfer_col()` method extracts columns using move semantics, transferring both numerical data and metadata (headers, encodings), while automatically updating indices to match the new configuration.

## Usage Example
```cpp
// Load CSV with ',' as separator, true for the header and AVX2_threaded for the method
auto df = CsvHandler::loadCsv("data.csv", ',', true, "AVX2_threaded"); // df is col-major

// Change layout by copy with Naive backend
df_row_major = df.change_layout("Naive");

// Change layout in-place with default backend
df.change_layout_inplace();

// Extract target column
auto y = df.transfer_col("target");

// Get i, j value (be careful of your layout and your indices)
auto val = df(i,j);
auto val = df.at(i*n + j);

// Display your df with encoded columns
df.display_raw();

// Display your df with decoded columns
df.display_decoded();
```

**To test it yourself**, you can also check the corresponding files: 
- [**Data.hpp**](include/Data/Data.hpp)
- [**Data.cpp**](include/Data/Data.cpp)
- [**Test file**](tests/tests_data.cpp)

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**II - Linear Algebra**](/docs/II_linalg.md).