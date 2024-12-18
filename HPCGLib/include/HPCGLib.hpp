#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <string>

// define number of iterations we want to have
#define num_bench_iter 10

enum class Implementation_Type {
    STRIPED,
    CSR,
    UNKNOWN
};

template <typename T>
class HPCG_functions {
    public:
        bool test_before_bench = true;
        std::string version_name = "unknown";
        const std::string ault_nodes = "41-44";
        // this string is used when small changes are benchmarked to see their effect
        std::string additional_parameters = "vanilla_version";

        Implementation_Type implementation_type = Implementation_Type::UNKNOWN;
        bool CG_implemented = false;
        bool MG_implemented = false;
        bool SymGS_implemented = false;
        bool SPMV_implemented = false;
        bool WAXPBY_implemented = false;
        bool Dot_implemented = false;

    // CG starts with having the data on the CPU
        virtual void compute_CG(sparse_CSR_Matrix<T> & A, std::vector<T> & b, std::vector<T> & x) = 0;
        
    // MG, SymGS, SPMV, WAXPBY and Dot have the data on the GPU already
        virtual void compute_MG(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_SymGS(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_SymGS(
            striped_Matrix<T> & A, // we pass A for the metadata
            T * striped_A_d, // the data of matrix A is already on the device
            int num_rows, int num_cols,
            int num_stripes, // the number of stripes in the striped matrix
            int * j_min_i, // this is a mapping for calculating the j of some entry i,j in the striped matrix
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        // this version supports CSR
        virtual void compute_SPMV(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            int * A_row_ptr_d, int * A_col_idx_d, T * A_values_d, // the matrix A is already on the device
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;
        
        // this version is for the striped matrix
        virtual void compute_SPMV(
            striped_Matrix<T>& A, //we only pass A for the metadata
            T * striped_A_d, // the matrix A is already on the device
            int num_rows, int num_cols, // these refer to the shape of the striped matrix
            int num_stripes, // the number of stripes in the striped matrix
            int * j_min_i_d, // this is a mapping for calculating the j of some entry i,j in the striped matrix
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_WAXPBY(
            T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
            T alpha, T beta
            ) = 0;

        virtual void compute_Dot(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) = 0;

        virtual void compute_Dot(
            striped_Matrix<T> & A, // we pass A for the metadata
            T * x_d, T * y_d, T * result_d // again: the vectors x, y and result are already on the device
        ) = 0;

        int getNumberOfIterations() const {
            return num_bench_iter;
    }
};

#endif // HPCGLIB_HPP