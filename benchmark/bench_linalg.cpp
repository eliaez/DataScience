#include <benchmark/benchmark.h>
#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"
#include <memory>
#include <random>

using namespace Linalg;

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

static void BM_MULT(benchmark::State& state) {

    const size_t N = state.range(0);
    const int backend_int = state.range(1);

    // Map backend_int and set it
    const std::string backend_names[] = {"Naive", "AVX2"};
    const std::string backend = backend_names[backend_int];
    Operations::set_backend(backend);

    Dataframe A = {N, N, true, gen_matrix(N,N)};
    Dataframe B = {N, N, false, gen_matrix(N,N)};

    for (auto _ : state) {
        Dataframe C = Operations::multiply(A, B);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
    state.SetLabel(get_backend());
}

static void GenerateArgs(benchmark::internal::Benchmark* b, int backend_int) {
    for (int size : {128, 256, 512, 1024}) {
        b->Args({size, backend_int});
    }
}

// Naive 
BENCHMARK(BM_MULT)
    ->Apply([](auto* b) { GenerateArgs(b, 0); })
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(2.0);

// AVX2
BENCHMARK(BM_MULT)
    ->Apply([](auto* b) { GenerateArgs(b, 1); })
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(2.0);