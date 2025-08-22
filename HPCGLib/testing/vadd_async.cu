#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>

// Constexpr vector size - huge number for benchmarking
constexpr size_t VECTOR_SIZE = 256 * 1024 * 256; // 64M elements
constexpr size_t BLOCK_SIZE = 256;
constexpr size_t GRID_SIZE = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Shared memory tile size for float buffering
constexpr size_t TILE_SIZE = 256; // Must match BLOCK_SIZE
constexpr size_t PIPELINE_DEPTH = 2; // Number of pipeline stages
constexpr size_t TILES_PER_BLOCK = 4;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

namespace cg = cooperative_groups;

// Simple synchronous vector addition kernel
__global__ void vectorAddSync(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// float buffered vector addition using memcpy_async and pipeline
__global__ void vectorAddPipelined(const float* A, const float* B, float* C, size_t n) {
    // Shared memory for float buffering
    __shared__ alignas(16) float sA_buffer[PIPELINE_DEPTH][TILE_SIZE];
    __shared__ alignas(16) float sB_buffer[PIPELINE_DEPTH][TILE_SIZE];

    // Create thread block group for cooperative operations
    auto block = cg::this_thread_block();

    // Pipeline object for managing async operations
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_DEPTH> pipe_state;
    auto pipeline = cuda::make_pipeline(block, &pipe_state);

    const size_t block_start = blockIdx.x * TILE_SIZE;
    const size_t elements_this_block = min(TILE_SIZE, n - block_start);

    // Only process if we have valid elements
    if (block_start >= n) return;

    // Issue async copy for the first tile
    pipeline.producer_acquire();

    // Use memcpy_async to copy from global to shared memory
    size_t copy_size = min(elements_this_block, TILE_SIZE) * sizeof(float);
    cuda::memcpy_async(block,
                      sA_buffer[0],
                      A + block_start,
                      copy_size,
                      pipeline);
    cuda::memcpy_async(block,
                      sB_buffer[0],
                      B + block_start,
                      copy_size,
                      pipeline);

    pipeline.producer_commit();

    // Wait for the data to arrive and compute
    pipeline.consumer_wait();
    block.sync();

    // Perform computation on loaded data
    int tid = threadIdx.x;
    size_t global_idx = block_start + tid;

    if (global_idx < n) {
        C[global_idx] = sA_buffer[0][tid] + sB_buffer[0][tid];
    }

    pipeline.consumer_release();
}

// Advanced pipelined vector addition with multiple tiles per block
__global__ void vectorAddMultiTilePipelined(const float* A, const float* B, float* C, size_t n) {
// Shared memory for pipeline buffering
    __shared__ alignas(16) float sA_buffer[PIPELINE_DEPTH][TILE_SIZE];
    __shared__ alignas(16) float sB_buffer[PIPELINE_DEPTH][TILE_SIZE];

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_DEPTH> pipe_state;
    auto pipeline = cuda::make_pipeline(block, &pipe_state);

    const int tid = threadIdx.x;
    const size_t block_start = blockIdx.x * TILE_SIZE * TILES_PER_BLOCK;

    size_t producer_idx = 0;
    size_t consumer_idx = 0;

    // Prefill the pipeline
    for (size_t tile_idx = 0; tile_idx < PIPELINE_DEPTH - 1; ++tile_idx) {
        const size_t tile_start = block_start + tile_idx * TILE_SIZE;

        if (tile_start >= n) break;

        const size_t elements_this_tile = min(TILE_SIZE, n - tile_start);
        const size_t copy_size = elements_this_tile * sizeof(float);

        pipeline.producer_acquire();

        cuda::memcpy_async(block,
                          sA_buffer[producer_idx],
                          A + tile_start,
                          copy_size,
                          pipeline);
        cuda::memcpy_async(block,
                          sB_buffer[producer_idx],
                          B + tile_start,
                          copy_size,
                          pipeline);

        producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
        pipeline.producer_commit();
    }



    // Process tiles with full pipeline overlap
    for (size_t tile_idx = 0; tile_idx < TILES_PER_BLOCK; ++tile_idx) {
        const size_t tile_start = block_start + tile_idx * TILE_SIZE;
        if (tile_start >= n) break;

        const size_t next_load_tile = tile_idx + (PIPELINE_DEPTH - 1);

        if (next_load_tile < TILES_PER_BLOCK) {
            const size_t next_load_tile_start = block_start + next_load_tile * TILE_SIZE;
            if (next_load_tile_start < n) {
                const size_t next_elements = min(TILE_SIZE, n - next_load_tile_start);
                const size_t next_copy_size = next_elements * sizeof(float);

                pipeline.producer_acquire();

                cuda::memcpy_async(block,
                                  sA_buffer[producer_idx],
                                  A + next_load_tile_start,
                                  next_copy_size,
                                  pipeline);
                cuda::memcpy_async(block,
                                  sB_buffer[producer_idx],
                                  B + next_load_tile_start,
                                  next_copy_size,
                                  pipeline);

                producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
                pipeline.producer_commit();
            }
        }

        // Wait for current tile and compute
        pipeline.consumer_wait();

        const size_t global_idx = tile_start + tid;
        if (global_idx < n) {
            C[global_idx] = sA_buffer[consumer_idx][tid] + sB_buffer[consumer_idx][tid];
        }
        consumer_idx = (consumer_idx + 1) % PIPELINE_DEPTH;
        pipeline.consumer_release();
    }

}

__global__ void vectorAddMultiTilePipelinedV2(const float* A, const float* B, float* C, size_t n) {
// Shared memory for pipeline buffering
    __shared__ alignas(16) float sA_buffer[PIPELINE_DEPTH][TILE_SIZE];
    __shared__ alignas(16) float sB_buffer[PIPELINE_DEPTH][TILE_SIZE];

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_DEPTH> pipe_state;
    auto pipeline = cuda::make_pipeline(block, &pipe_state);

    const int tid = threadIdx.x;
    const size_t block_start = blockIdx.x * TILE_SIZE * TILES_PER_BLOCK;

    size_t producer_idx = 0;
    size_t consumer_idx = 0;

    // Prefill the pipeline
    for (size_t tile_idx = 0; tile_idx < PIPELINE_DEPTH - 1; ++tile_idx) {
        const size_t tile_start = block_start + tile_idx * TILE_SIZE;

        if (tile_start >= n) break;

        const size_t elements_this_tile = min(TILE_SIZE, n - tile_start);
        const size_t copy_size = elements_this_tile * sizeof(float);

        pipeline.producer_acquire();

        cuda::memcpy_async(block,
                          sA_buffer[producer_idx],
                          A + tile_start,
                          copy_size,
                          pipeline);
        cuda::memcpy_async(block,
                          sB_buffer[producer_idx],
                          B + tile_start,
                          copy_size,
                          pipeline);

        producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
        pipeline.producer_commit();
    }



    // Process tiles with full pipeline overlap
    for (size_t tile_idx = 0; tile_idx < TILES_PER_BLOCK; ++tile_idx) {
        const size_t tile_start = block_start + tile_idx * TILE_SIZE;
        if (tile_start >= n) break;

        const size_t next_load_tile = tile_idx + (PIPELINE_DEPTH - 1);

        if (next_load_tile < TILES_PER_BLOCK) {
            const size_t next_load_tile_start = block_start + next_load_tile * TILE_SIZE;
            if (next_load_tile_start < n) {
                const size_t next_elements = min(TILE_SIZE, n - next_load_tile_start);
                const size_t next_copy_size = next_elements * sizeof(float);

                pipeline.producer_acquire();

                cuda::memcpy_async(block,
                                  sA_buffer[producer_idx],
                                  A + next_load_tile_start,
                                  next_copy_size,
                                  pipeline);
                cuda::memcpy_async(block,
                                  sB_buffer[producer_idx],
                                  B + next_load_tile_start,
                                  next_copy_size,
                                  pipeline);

                producer_idx = (producer_idx + 1) % PIPELINE_DEPTH;
                pipeline.producer_commit();
            }
        }

        // Wait for current tile and compute
        pipeline.consumer_wait();

        const size_t global_idx = tile_start + tid;
        if (global_idx < n) {
            C[global_idx] = sA_buffer[consumer_idx][tid] + sB_buffer[consumer_idx][tid];
        }
        consumer_idx = (consumer_idx + 1) % PIPELINE_DEPTH;
        pipeline.consumer_release();
    }

}

// Function to generate input vectors using simple patterns
void generateInputVectors(std::vector<float>& A, std::vector<float>& B) {
    A.resize(VECTOR_SIZE);
    B.resize(VECTOR_SIZE);

    std::cout << "Generating input vectors with patterns..." << std::endl;

    #pragma omp parallel for
    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        A[i] = i * 1.1 + 0.5;           // Pattern: i*1.1 + 0.5
        B[i] = i * 0.7 + 1.3;           // Pattern: i*0.7 + 1.3
    }

    std::cout << "Input vectors generated successfully." << std::endl;
}

// Function to generate expected output vector (CPU reference)
void generateExpectedOutput(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& expected) {
    expected.resize(VECTOR_SIZE);

    std::cout << "Computing expected output on CPU..." << std::endl;

    #pragma omp parallel for
    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        expected[i] = A[i] + B[i];
    }

    std::cout << "Expected output computed." << std::endl;
}

// Function to compare GPU output with expected output
bool compareResults(const std::vector<float>& gpu_result,
                   const std::vector<float>& expected,
                   float tolerance = 1e-9) {
    std::cout << "Comparing GPU results with expected output..." << std::endl;

    bool passed = true;
    size_t error_count = 0;
    const size_t max_errors_to_show = 10;

    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        float diff = std::abs(gpu_result[i] - expected[i]);
        if (diff > tolerance) {
            if (error_count < max_errors_to_show) {
                std::cout << "Mismatch at index " << i << ": GPU=" << gpu_result[i]
                         << ", Expected=" << expected[i] << ", Diff=" << diff << std::endl;
            }
            error_count++;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "✓ All results match! Test PASSED." << std::endl;
    } else {
        std::cout << "✗ Found " << error_count << " mismatches (" << (static_cast<float>(error_count)*100.f)/static_cast<float>(VECTOR_SIZE) << " %). Test FAILED." << std::endl;
    }

    return passed;
}

// Benchmark helper function
float runKernelBenchmark(const char* name, auto kernel_func,
                         const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& C, size_t grid_size) {
    std::cout << "\n=== " << name << " ===" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t bytes = VECTOR_SIZE * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    // Warm up
    kernel_func<<<grid_size, BLOCK_SIZE>>>(d_A, d_B, d_C, VECTOR_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    kernel_func<<<grid_size, BLOCK_SIZE>>>(d_A, d_B, d_C, VECTOR_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end - start).count();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << name << " time: " << duration << " ms" << std::endl;
    return duration;
}

// Main benchmark function
int main() {
    std::cout << "CUDA Vector Addition Benchmark - Pipeline and memcpy_async" << std::endl;
    std::cout << "Vector size: " << VECTOR_SIZE << " elements" << std::endl;
    std::cout << "Memory size: " << (VECTOR_SIZE * sizeof(float) / 1024.0 / 1024.0) << " MB per vector" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << " elements" << std::endl;
    std::cout << "Pipeline depth: " << PIPELINE_DEPTH << std::endl;
    std::cout << "Shared memory per block: " << (PIPELINE_DEPTH * 2 * TILE_SIZE * sizeof(float) / 1024.0) << " KB" << std::endl;

    // Check GPU compute capability (memcpy_async requires 8.0+)
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;

    if (prop.major < 8) {
        std::cout << "WARNING: memcpy_async requires Compute Capability 8.0+. Some features may not work optimally." << std::endl;
    }

    // Generate input vectors
    std::vector<float> A, B;
    generateInputVectors(A, B);

    // Generate expected output
    std::vector<float> expected;
    generateExpectedOutput(A, B, expected);

    // Prepare output vectors
    std::vector<float> C_sync(VECTOR_SIZE);
    std::vector<float> C_pipeline(VECTOR_SIZE);
    std::vector<float> C_multi_tile(VECTOR_SIZE);
    std::vector<float> C_multi_tile_v2(VECTOR_SIZE);

    // Warm-up
    runKernelBenchmark("Synchronous", vectorAddSync, A, B, C_sync, GRID_SIZE);

    // Run benchmarks
    float sync_time = runKernelBenchmark("Synchronous", vectorAddSync, A, B, C_sync, GRID_SIZE);
    float pipeline_time = runKernelBenchmark("Basic Pipeline", vectorAddPipelined, A, B, C_pipeline, GRID_SIZE);

    size_t multi_tile_grid = (VECTOR_SIZE + TILE_SIZE * TILES_PER_BLOCK - 1) / (TILE_SIZE * TILES_PER_BLOCK);
    float multi_tile_time = runKernelBenchmark("Multi-Tile Pipeline", vectorAddMultiTilePipelined, A, B, C_multi_tile, multi_tile_grid);

    size_t multi_tile_v2_grid = (VECTOR_SIZE + TILE_SIZE * TILES_PER_BLOCK - 1) / (TILE_SIZE * TILES_PER_BLOCK);
    float multi_tile_v2_time = runKernelBenchmark("Multi-Tile Pipeline V2", vectorAddMultiTilePipelinedV2, A, B, C_multi_tile_v2, multi_tile_v2_grid);


    // Compare results
    std::cout << "\n=== Results Verification ===" << std::endl;
    std::cout << "Synchronous: ";
    bool sync_passed = compareResults(C_sync, expected);

    std::cout << "Basic Pipeline: ";
    bool pipeline_passed = compareResults(C_pipeline, expected);

    std::cout << "Multi-Tile Pipeline: ";
    bool multi_tile_passed = compareResults(C_multi_tile, expected);

    //std::cout << "Multi-Tile Pipeline: ";
    //bool multi_tile_v2_passed = compareResults(C_multi_tile_v2, expected);


    // Performance summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Synchronous time:       " << sync_time << " ms" << std::endl;
    std::cout << "Basic Pipeline time:    " << pipeline_time << " ms" << std::endl;
    std::cout << "Multi-Tile time:        " << multi_tile_time << " ms" << std::endl;
    //std::cout << "Multi-Tile V2 time:     " << multi_tile_v2_time << " ms" << std::endl;

    std::cout << "\nSpeedup vs Synchronous:" << std::endl;
    std::cout << "Basic Pipeline:   " << (sync_time / pipeline_time) << "x" << std::endl;
    std::cout << "Multi-Tile:       " << (sync_time / multi_tile_time) << "x" << std::endl;
    //std::cout << "Multi-Tile V2:    " << (sync_time / multi_tile_v2_time) << "x" << std::endl;

    // Bandwidth calculation
    float bytes_accessed = 3.0 * VECTOR_SIZE * sizeof(float);
    std::cout << "\nMemory Bandwidth:" << std::endl;
    std::cout << "Synchronous:      " << (bytes_accessed / 1e9) / (sync_time / 1000.0) << " GB/s" << std::endl;
    std::cout << "Basic Pipeline:   " << (bytes_accessed / 1e9) / (pipeline_time / 1000.0) << " GB/s" << std::endl;
    std::cout << "Multi-Tile:       " << (bytes_accessed / 1e9) / (multi_tile_time / 1000.0) << " GB/s" << std::endl;
    //std::cout << "Multi-Tile V2:    " << (bytes_accessed / 1e9) / (multi_tile_v2_time / 1000.0) << " GB/s" << std::endl;

    return (sync_passed && pipeline_passed && multi_tile_passed) ? 0 : 1;
}