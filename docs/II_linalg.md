# II - Linear Algebra
Before implementing any ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to achieve better performance, so we will implement 3 different approaches and add 2 external libraries to compare. Thus, to enable backend selection and operation dispatching, the [**Linalg**](include/Linalg/Linalg.hpp) namespace provides a unified interface. This abstraction layer enables transparent backend switching without altering function signatures.

## Backends
### Implemented:
- **`Naive`:**
    Basic C++ implementation using standard loops and operations. Still with basic optimization: column-major storage for cache-friendly operations. Serves as a baseline for performance comparison.

- **`AVX2`:**
    Implementation using AVX2 SIMD instructions for vectorized operations, processing 256-bit registers (4 doubles simultaneously) and operating on blocks where applicable.

- **`AVX2 Threaded`:**
    Multi-threaded AVX2 implementation combining SIMD vectorization with parallel processing across CPU cores using std::thread and operating on blocks where applicable for maximum performance.

### Externals:
- **`Eigen`:** 
    C++ template library for linear algebra: matrices, vectors, numerical solvers and related algorithms.<br>
    For further details, see https://gitlab.com/libeigen/eigen.

- **`MKL`:**
    Intel's optimized Math Kernel Library for high-performance computing. <br> 
    For further details, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html.

## Architecture of `Linalg`

**Implementation note**: `Operations` provides a validated `Dataframe` interface wrapping performance-optimized backends that operate on raw data. Backend headers are kept internal (`src/Linalg/backends/`) for encapsulation while remaining accessible for testing.


#### Backend System
The `Operations` class manages backend selection through compile-time feature detection and runtime dispatch:

```cpp
enum class Backend {
    NAIVE,           // Always available
    AVX2,            // If __AVX2__ defined
    AVX2_THREADED,   // If __AVX2__ defined
    EIGEN,           // Always available
    MKL,             // If USE_MKL defined
    AUTO             // Selects best available
};
```

Backend availability is determined at compile-time via preprocessor guards, ensuring only supported implementations are included. The `AUTO` mode defaults to `AVX2_THREADED` (if available) or `NAIVE` otherwise.

#### Dispatch Mechanism
Operations are routed through a two-layer architecture: first through the Linalg PIMPL-like interface, then to the corresponding backend using a macro-based dispatch (`DISPATCH_BACKEND`) that expands to switch statements, eliminating virtual function overhead:

```cpp
// User calls (high-level)
auto result = Linalg::Operations::multiply(df1, df2);

// Internally dispatches to (low-level)
Linalg::Operations::Impl::multiply_impl(
    raw_data1, raw_data2, rows1, cols1, rows2, cols2, layout1, layout2
);

// Then dispatches to AVX2 threaded backend (default)
Linalg::AVX2_threaded::multiply(
    raw_data1, raw_data2, rows1, cols1, rows2, cols2
);
```

## Core Operations

- **Matrix arithmetic**: `sum()`, `multiply()` 
- **Transformations**: `transpose()`, `inverse()`
- **Utilities**: `determinant()` with triangular matrix detection (scalar or AVX2-optimized) with a LU decomposition fallback

#### Layout Optimization
Operations automatically handle layout conversions when needed. For instance, `determinant()` converts to row-major if the input is column-major to optimize cache access patterns during triangular checks.

## Usage Example

```cpp
// Set backend explicitly
Linalg::Operations::set_backend("AVX2_threaded");
// Or
Linalg::Operations::set_backend(Backend::AVX2_THREADED);

// Perform operations
auto df_t = Linalg::Operations::transpose(df);
auto df_sum = Linalg::Operations::sum(df1, df2);
auto df_mult = Linalg::Operations::multiply(df1, df2);
auto df_inv = Linalg::Operations::inverse(df);
```

**To test it yourself**, you can also check the corresponding files: 
- [**Linalg.hpp**](include/Linalg/Linalg.hpp)
- [**Linalg.cpp**](src/Linalg/Linalg.cpp)
- [**Test folder**](tests/)

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**III - Linear Algebra Benchmark**](/docs/III_benchmark.md).
