# Machine Learning From Scratch
Educational project implementing fundamental machine learning and regression algorithms in C++ without relying on external ML libraries.

## Linear Algebra
Before implementing ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to get better performances, so we'll try 4 different approachs to compare: 

### Naive
Basic C++ implementation using standard loops and operations. Still with basic optimization: column-major storage for cache-friendly operations. Serves as baseline for performance comparison.

### AVX2
Implementation using AVX2 SIMD instructions for vectorized operations, processing 256-bit registers ie 4 doubles simultaneously.

### AVX2 Threaded
Multi-threaded AVX2 implementation combining SIMD vectorization with parallel processing across CPU cores for maximum performance.

### Eigen
Industry-standard C++ template library.
