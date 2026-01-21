#include <benchmark/benchmark.h>
#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"
#include <memory>
#include <random>

using namespace Linalg;

std::vector<double> gen_matrix(size_t rows, size_t cols);

static void BM_MULT(benchmark::State& state) {

    const size_t N = state.range(0);
    const int backend_int = state.range(1);

    // Map backend_int and set it
    const std::string backend_names[] = {"Naive", "AVX2", "Eigen", "MKL", "AVX2_threaded"};
    const std::string backend = backend_names[backend_int];
    Operations::set_backend(backend);

    Dataframe A;
    if (backend == "Eigen" || backend == "MKL") A = {N, N, false, gen_matrix(N,N)};
    else A = {N, N, true, gen_matrix(N,N)};

    Dataframe B = {N, N, false, gen_matrix(N,N)};

    for (auto _ : state) {
        Dataframe C = Operations::multiply(A, B);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
    state.SetLabel(get_backend());
}

static void BM_INV(benchmark::State& state) {

    const size_t N = state.range(0);
    const int backend_int = state.range(1);

    // Map backend_int and set it
    const std::string backend_names[] = {"Naive", "AVX2", "Eigen", "MKL", "AVX2_threaded"};
    const std::string backend = backend_names[backend_int];
    Operations::set_backend(backend);

    Dataframe A = {N, N, false, gen_matrix(N,N)};

    for (auto _ : state) {
        Dataframe B = Operations::inverse(A);
        benchmark::DoNotOptimize(B);
        benchmark::ClobberMemory();
    }
    state.SetLabel(get_backend());
}

static void GenerateArgs(benchmark::Benchmark* b, int backend_int) {
    for (int size : {64, 128, 192, 256, 384, 512, 768, 1024}) { // {64, 128, 192, 256, 384, 512, 768, 1024} {768, 1024, 1536, 2048, 3072, 4096}
        b->Args({size, backend_int});
    }
}

// ------------------------ MULT ------------------------
BENCHMARK(BM_MULT)
    ->Apply([](auto* b) { 
        GenerateArgs(b, 0);  // Naive
#ifdef __AVX2__
        GenerateArgs(b, 1);  // AVX2
        GenerateArgs(b, 4);  // AVX2_threaded
#endif
        GenerateArgs(b, 2);  // Eigen
#ifdef USE_MKL
        GenerateArgs(b, 3);  // MKL
#endif
    })
    /*->MinTime(2.0) // Only for final bench
    ->Repetitions(10)
    ->ReportAggregatesOnly(true) 
    ->DisplayAggregatesOnly(true)*/
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------ INV ------------------------
BENCHMARK(BM_INV)
    ->Apply([](auto* b) { 
        GenerateArgs(b, 0);  // Naive
#ifdef __AVX2__
        GenerateArgs(b, 1);  // AVX2
        GenerateArgs(b, 4);  // AVX2_threaded
#endif
        GenerateArgs(b, 2);  // Eigen
#ifdef USE_MKL
        GenerateArgs(b, 3);  // MKL
#endif
    })
    /*->MinTime(2.0) // Only for final bench
    ->Repetitions(10)
    ->ReportAggregatesOnly(true) 
    ->DisplayAggregatesOnly(true)*/
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

