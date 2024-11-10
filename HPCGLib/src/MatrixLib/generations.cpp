#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include "MatrixLib/generations.hpp"
#include <vector>
#include <iostream>
#include <cassert>
#include <utility> // for std::pair

std::pair<sparse_CSR_Matrix<double>, std::vector<double>> generate_HPCG_Problem(int nx, int ny, int nz){

    int num_rows = nx * ny * nz;
    int num_cols = nx * ny * nz;

    int nnz = 0;

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

    // Create the sparse matrix
    sparse_CSR_Matrix<double> A(nx, ny, nz, nnz, MatrixType::Stencil_3D27P, values, row_ptr, col_idx);

    return std::make_pair(A, y);
}

std::pair<sparse_CSR_Matrix<double>, std::vector<int>> generate_coarse_HPCG_Problem(int nxf, int nyf, int nzf){

    assert(nxf % 2 == 0);
    assert(nyf % 2 == 0);
    assert(nzf % 2 == 0);

    int nxc = nxf / 2;
    int nyc = nyf / 2;
    int nzc = nzf / 2;

    int num_fine_rows = nxf * nyf * nzf;
    int num_fine_cols = nxf * nyf * nzf;

    std::vector<int>f2c_op(num_fine_rows, 0.0);

    for (int izc=0; izc<nzc; ++izc) {
        int izf = 2*izc;
        for (int iyc=0; iyc<nyc; ++iyc) {
            int iyf = 2*iyc;
            for (int ixc=0; ixc<nxc; ++ixc) {
                int ixf = 2*ixc;
                int currentCoarseRow = izc*nxc*nyc+iyc*nxc+ixc;
                int currentFineRow = izf*nxf*nyf+iyf*nxf+ixf;
                f2c_op[currentCoarseRow] = currentFineRow;
            } // end iy loop
        } // end even iz if statement
    } // end iz loop

    // sparse_CSR_Matrix<double> Ac;
    // std::vector<double> yc;

    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> cProblem = generate_HPCG_Problem(nxc, nyc, nzc);
    return std::make_pair(cProblem.first, f2c_op);

}

// Also make a generation for the banded matrix in here (at some point)

std::vector<double> generate_random_vector(int size, int seed){
    std::vector<double> vec(size, 0.0);
    srand(seed);
    for (int i = 0; i < size; i++){
        vec[i] = (double)rand() / RAND_MAX;
    }
    return vec;

}