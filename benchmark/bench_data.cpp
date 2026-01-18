#include <benchmark/benchmark.h>
#include "Data/Data.hpp"
#include <memory>
#include <random>


std::vector<double> gen_matrix(size_t rows, size_t cols) {
    std::vector<double> mat(rows*cols);

    // Generate mat
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (auto& val : mat) {
        val = dis(gen);
    }
    return mat;
}

static void BM_TRANSPOSE_IN(benchmark::State& state) {

    const size_t N = state.range(0);
    const int backend_int = state.range(1);

    // Map backend_int
    const std::string backend_names[] = {"Naive", "AVX2", "Eigen", "MKL", "AVX2_threaded"};
    const std::string backend = backend_names[backend_int];

    Dataframe A = {N, N, false, gen_matrix(N,N)};

    for (auto _ : state) {
        A.change_layout_inplace(backend);
        benchmark::DoNotOptimize(A);
        benchmark::ClobberMemory();
    }
    state.SetLabel(backend);
}

static void BM_TRANSPOSE(benchmark::State& state) {

    const size_t N = state.range(0);
    const int backend_int = state.range(1);

    // Map backend_int
    const std::string backend_names[] = {"Naive", "AVX2", "Eigen", "MKL", "AVX2_threaded"};
    const std::string backend = backend_names[backend_int];

    Dataframe A = {N, N, false, gen_matrix(N,N)};
    Dataframe B;

    for (auto _ : state) {
        B = A.change_layout(backend);
        benchmark::DoNotOptimize(B);
        benchmark::ClobberMemory();
    }
    state.SetLabel(backend);
}

static void GenerateArgs(benchmark::Benchmark* b) {
    
    #if defined(__AVX2__) && defined(USE_MKL)
        std::vector<int> backend_opt = {0, 1, 2, 3, 4};    // Naive, AVX2, Eigen, MKL, AVX2_threaded
    #elif defined(__AVX2__)
        std::vector<int> backend_opt = {0, 1, 2, 4};       // {0, 1, 2, 4} Naive, AVX2, Eigen, AVX2_threaded
    #elif defined(USE_MKL)
        std::vector<int> backend_opt = {0, 2, 3};       // Naive, Eigen, MKL 
    #else
        std::vector<int> backend_opt = {0, 2};          // Naive, Eigen 
    #endif

    for (int backend : backend_opt) {
        for (int size : {128, 256, 512, 1024, 2048}) { 
            b->Args({size, backend});
        }
    }
}

BENCHMARK(BM_TRANSPOSE_IN)
    ->Apply(GenerateArgs)
/*    ->MinTime(2.0) // Only for final bench
    ->Repetitions(10)
    ->ReportAggregatesOnly(true) 
    ->DisplayAggregatesOnly(true)*/
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_TRANSPOSE)
    ->Apply(GenerateArgs)
/*    ->MinTime(2.0) // Only for final bench
    ->Repetitions(10)
    ->ReportAggregatesOnly(true) 
    ->DisplayAggregatesOnly(true)*/
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);