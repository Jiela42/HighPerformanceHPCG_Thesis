
// this imports are from AMGX (DO NOT TOUCH)
#include "matrix.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "multiply.h"
#include "thrust/fill.h"

// here are my imports
#include <vector>
#include <tuple>

// First we have all the generations we might need

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> generate_HPCG_problem(int nx, int ny, int nz)
{
    
    // this is a copy of a generation method in HighPerformanceHPCG_thesis

    int num_rows = nx * ny * nz;
    int num_cols = nx * ny * nz;

    int nnz = 0;
    // printf("Generating problem with %d rows, %d cols, %d nnz\n", num_rows, num_cols, nnz);

    // return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<double>(), std::vector<double>());

    std::vector<int> row_ptr(num_rows + 1, 0);
    std::vector<int> nnz_per_row(num_rows);
    std::vector<int> col_idx;
    std::vector<double> values;
    std::vector<double> y(num_rows, 0.0);

    std::vector<std::vector<int>> col_idx_per_row(num_rows);
    std::vector<std::vector<double>> values_per_row(num_rows);

    for(int ix = 0; ix < nx; ix++){
        for(int iy = 0; iy < ny; iy++){
            for(int iz = 0; iz < nz; iz++){

                int i = ix + nx * iy + nx * ny * iz;
                int nnz_i = 0;

                for (int sz = -1; sz < 2; sz++){
                    if(iz + sz > -1 && iz + sz < nz){
                        for(int sy = -1; sy < 2; sy++){
                            if(iy + sy > -1 && iy + sy < ny){
                                for(int sx = -1; sx < 2; sx++){
                                    if(ix + sx > -1 && ix + sx < nx){
                                        int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                        if(i == j){
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(26.0);
                                        } else {
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(-1.0);
                                        }
                                            nnz_i++;
                                            nnz++;
                                    }
                                }
                            }
                        }
                    }
                }
                nnz_per_row[i] = nnz_i;
                y[i] = 26.0 - nnz_i;
            }
        }
    }

    for (int i = 0; i < num_rows; i++){
        row_ptr[i + 1] = row_ptr[i] + nnz_per_row[i];

        for (int j = 0; j < nnz_per_row[i]; j++){
            col_idx.push_back(col_idx_per_row[i][j]);
            values.push_back(values_per_row[i][j]);
        }
    }
    // printf("Generated problem with %d rows, %d cols, %d nnz\n", num_rows, num_cols, nnz);
    return std::make_tuple(row_ptr, col_idx, values, y);
}

void registerParameters();

void spmv_example(amgx::Resources& res)
{
    // TemplateConfig parameters:
    // calculate on device
    // double storage for matrix values
    // double storage for vector values
    // integer for indices values
    //
    // see include/basic_types.h for details

    // generate problem
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> problem = generate_HPCG_problem(8, 8, 8);

    std::vector<int>& row_ptr = std::get<0>(problem);
    std::vector<int>& col_idx = std::get<1>(problem);
    std::vector<double>& values = std::get<2>(problem);
    std::vector<double>& y2 = std::get<3>(problem);


    typedef amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> TConfig; // Type for spmv calculation
    typedef amgx::Vector<amgx::TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> VVector_h; // vector type to retrieve result

    amgx::Matrix<TConfig> A;
    amgx::Vector<TConfig> x;
    amgx::Vector<TConfig> y;
    A.setResources(&res);
    x.setResources(&res);
    y.setResources(&res);

    int nrows = 5;
    int nnz = 13;
    A.resize(nrows, nrows, nnz, 1);
    x.resize(nrows);
    y.resize(nrows);
    amgx::thrust::fill(y.begin(), y.end(), 0.);
    
    // matrix row offsets
    std::vector<int> row_offsets_h = {0, 2, 5, 7, 10, 13};
    A.row_offsets.assign(row_offsets_h.begin(), row_offsets_h.end());
    
    // matrix colums indices 
    std::vector<int> column_indices_h = {0, 2, 1, 3, 4, 0, 2, 2, 3, 4, 0, 2 ,4};
    A.col_indices.assign(column_indices_h.begin(), column_indices_h.end());
    
    // matrix values
    std::vector<double> matrix_values_h(nnz);
    for (auto &v : matrix_values_h) v = ((double)random())/RAND_MAX;
    A.values.assign(matrix_values_h.begin(), matrix_values_h.end());
    //set matrix "completeness" flag
    A.set_initialized(1);

    // vector values
    std::vector<double> x_h(nrows);
    for (auto &v : x_h) v = ((double)random())/RAND_MAX;
    x.assign(x_h.begin(), x_h.end());

    // AMGX multiply 
    amgx::multiply(A, x, y);

    // get the result to host
    VVector_h y_res_h = y;

    // reference check
    std::vector<double> y_res_ref(nrows, 0.);
    bool err_found = false;
    for (int r = 0; r < nrows; r++)
    {
        double y_res_ref = 0.;
        for (int c = row_offsets_h[r]; c < row_offsets_h[r + 1]; c++)
        {
            y_res_ref += matrix_values_h[c]*x_h[column_indices_h[c]];
        }
        if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
        {
            printf("Difference in row %d: reference: %f, AMGX: %f\n", r, y_res_ref, y_res_h[r]);
            err_found = true;
        }
    }

    if (!err_found)
        printf("Done!\n");
}

int main(int argc, char* argv[])
{
    // Initialization
    cudaSetDevice(0);
    // register required AMGX parameters 
    registerParameters();
    
    // resources object
    amgx::Resources res;

    // make sure we perform any AMGX functionality within Resources lifetime
    spmv_example(res);
    
    return 0;
}


// Routine to register some of the AMGX parameters manually
// Typically if you want to use AMGX solver you should call core::initialize() which will cover this initialization, 
// however for spmv alone there are only few parameters needed and no need to initialize AMGX core solvers.
void registerParameters()
{
    using namespace amgx;
    std::vector<int> bool_flag_values;
    bool_flag_values.push_back(0);
    bool_flag_values.push_back(1);
    //Register Exception Handling Parameter
    AMG_Config::registerParameter<int>("exception_handling", "a flag that forces internal exception processing instead of returning error codes(1:internal, 0:external)", 0, bool_flag_values);
    //Register System Parameters (memory pools)
    AMG_Config::registerParameter<size_t>("device_mem_pool_size", "size of the device memory pool in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_consolidation_pool_size", "size of the device memory pool for root partition in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_max_alloc_size", "maximum size of a single allocation in the device memory pool in bytes", 20 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_factor", "over allocation for large buffers (in %% -- a value of X will lead to 100+X%% allocations)", 10);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_threshold", "buffers smaller than that threshold will NOT be scaled", 16 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_size_limit", "size of the device memory pool in bytes. 0 - no limit", 0);
    //Register System Parameters (asynchronous framework)
    AMG_Config::registerParameter<int>("num_streams", "number of additional CUDA streams / threads used for async execution", 0);
    AMG_Config::registerParameter<int>("serialize_threads", "flag that enables thread serialization for debugging <0|1>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("high_priority_stream", "flag that enables high priority CUDA stream <0|1>", 0, bool_flag_values);
    //Register System Parameters (in distributed setting)
    std::vector<std::string> communicator_values;
    communicator_values.push_back("MPI");
    communicator_values.push_back("MPI_DIRECT");
    AMG_Config::registerParameter<std::string>("communicator", "type of communicator <MPI|MPI_DIRECT>", "MPI");
    std::vector<ViewType> viewtype_values;
    viewtype_values.push_back(INTERIOR);
    viewtype_values.push_back(OWNED);
    viewtype_values.push_back(FULL);
    viewtype_values.push_back(ALL);
    AMG_Config::registerParameter<ViewType>("separation_interior", "separation for latency hiding and coloring/smoothing <ViewType>", INTERIOR, viewtype_values);
    AMG_Config::registerParameter<ViewType>("separation_exterior", "limit of calculations for coloring/smoothing <ViewType>", OWNED, viewtype_values);
    AMG_Config::registerParameter<int>("min_rows_latency_hiding", "number of rows at which to disable latency hiding, negative value means latency hiding is completely disabled", -1);
    AMG_Config::registerParameter<int>("matrix_halo_exchange", "0 - No halo exchange on lower levels, 1 - just diagonal values, 2 - full", 0);
    AMG_Config::registerParameter<std::string>("solver", "", "");
    AMG_Config::registerParameter<int>("verbosity_level", "verbosity level for output, 3 - custom print-outs <0|1|2|3>", 3);   
}
