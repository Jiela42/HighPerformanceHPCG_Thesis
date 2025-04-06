#ifndef HPCGLIB_HPP
#define HPCGLIB_HPP

#include "MatrixLib/sparse_CSR_Matrix.hpp"
#include <vector>
#include <string>

#include "UtilLib/cuda_utils.hpp"
#include "cuda_runtime.h"

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
        bool test_before_bench = false;
        std::string version_name = "unknown";
        const std::string ault_nodes = "GH200";
        // this string is used when small changes are benchmarked to see their effect
        std::string additional_parameters = "vanilla_version";

        bool doPreconditioning = false;

        Implementation_Type implementation_type = Implementation_Type::UNKNOWN;
        bool norm_based = false;
        bool CG_implemented = false;
        bool MG_implemented = false;
        bool SymGS_implemented = false;
        bool SPMV_implemented = false;
        bool WAXPBY_implemented = false;
        bool Dot_implemented = false;

        bool CG_file_based_tests_passed = false;
        bool MG_file_based_tests_passed = false;

        void add_additional_parameters(std::string another_parameter) {
            additional_parameters += "_" + another_parameter;
        }

        double get_CGTolerance() const {
            return CG_tolerance;
        }
        void set_CGTolerance(double new_tolerance) {
            CG_tolerance = new_tolerance;
        }

        int get_maxCGIters() const {
            return max_CG_iterations;
        }
        void set_maxCGIters(int new_max_iters) {
            max_CG_iterations = new_max_iters;
        }
        void set_maxSymGSIters(int new_max_iters) {
            max_SymGS_iterations = new_max_iters;
        }
        int get_maxSymGSIters() const {
            return max_SymGS_iterations;
        }


    // CG starts with having the data on the CPU
        virtual void compute_CG(
            striped_Matrix<T> & A,
            T * b_d, T * x_d,
            int & n_iters, T& normr, T& normr0) = 0;
    
        void compute_CG(
            sparse_CSR_Matrix<T> & A,
            T * b_d, T * x_d,
            int & n_iters, T& normr, T& normr0
        ) {
            striped_Matrix<T>* striped_A = A.get_Striped();
            compute_CG(*striped_A, b_d, x_d, n_iters, normr, normr0);
        }
        
    // MG, SymGS, SPMV, WAXPBY and Dot have the data on the GPU already
        virtual void compute_MG(
            striped_Matrix<T> & A,
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_SymGS(
            sparse_CSR_Matrix<T> & A,
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_SymGS(
            striped_Matrix<T> & A,
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        // this version supports CSR
        // Ax = y
        virtual void compute_SPMV(
            sparse_CSR_Matrix<T> & A,
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;
        
        // this version is for the striped matrix
        virtual void compute_SPMV(
            striped_Matrix<T>& A, //we only pass A for the metadata
            T * x_d, T * y_d // the vectors x and y are already on the device
            ) = 0;

        virtual void compute_WAXPBY(
            sparse_CSR_Matrix<T> & A, // we pass A for the metadata
            T * x_d, T * y_d, T * w_d, // the vectors x, y and w are already on the device
            T alpha, T beta
            ) = 0;
        
        virtual void compute_WAXPBY(
            striped_Matrix<T> & A, // we pass A for the metadata
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

    double L2_norm_for_SymGS(
        striped_Matrix<T> & A,
        double * x,
        double * y
    ){
        global_int_t num_rows = A.get_num_rows();

        // Allocate memory for Ax on the device
        double *Ax;
        double *result;
        CHECK_CUDA(cudaMalloc(&Ax, num_rows * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&result, 1 * sizeof(T)));
    
        CHECK_CUDA(cudaMemset(Ax, 0, num_rows * sizeof(T)));
        CHECK_CUDA(cudaMemset(result, 0, 1 * sizeof(T)));
    
        // Perform matrix-vector multiplication: Ax = A * x
    
        compute_SPMV(A, x, Ax);
        compute_WAXPBY(A, Ax, y, Ax, 1.0, -1.0);
        compute_Dot(A, Ax, Ax, result);
    
        // copy result over
        T result_h;
        CHECK_CUDA(cudaMemcpy(&result_h, result, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    
    
        // Free device memory
        CHECK_CUDA(cudaFree(Ax));
        CHECK_CUDA(cudaFree(result));
    
        return static_cast<double>(std::sqrt(result_h));
    }

        double getSymGS_rrNorm(int nx, int ny, int nz){
            // The relative residual norm for 2x2x2 for random initialization is 0.01780163474925223
            // The relative residual norm for 4x4x4 for random initialization is 0.06265609442727489
            // The relative residual norm for 8x8x8 for random initialization is 0.06901259891475327
            // The relative residual norm for 16x16x16 for random initialization is 0.1093031516812343
            // The relative residual norm for 32x32x32 for random initialization is 0.2059749871712245
            // The relative residual norm for 64x64x64 for random initialization is 0.3318764588131313
            // The relative residual norm for 128x64x64 for random initialization is 0.3686739441774081
            // The relative residual norm for 128x128x64 for random initialization is 0.415239353783707
            // The relative residual norm for 128x128x128 for random initialization is 0.4761695919404465
            // The relative residual norm for 256x128x128 for random initialization is 0.5161210410022423

            if (nx == 2 && ny == 2 && nz == 2) {
                return 0.01780163474925223;
            } else if (nx == 4 && ny == 4 && nz == 4) {
                return 0.06265609442727489;
            } else if (nx == 8 && ny == 8 && nz == 8) {
                return 0.06901259891475327;
            } else if (nx == 16 && ny == 16 && nz == 16) {
                return 0.1093031516812343;
            } else if (nx == 24 && ny == 24 && nz == 24) {
                return 0.161818728347593;
            } else if (nx == 32 && ny == 32 && nz == 32) {
                return 0.2059749871712245;
            } else if (nx == 64 && ny == 64 && nz == 64) {
                return 0.3318764588131313;
            } else if (nx == 128 && ny == 64 && nz == 64) {
                return 0.3686739441774081;
            } else if (nx == 128 && ny == 128 && nz == 64) {
                return 0.415239353783707;
            } else if (nx == 128 && ny == 128 && nz == 128) {
                return 0.4761695919404465;
            } else if (nx == 256 && ny == 128 && nz == 128) {
                return 0.5161210410022423;
            }else if (nx == 256 && ny == 256 && nz == 128) {
                return 0.5649574838245627;
            } else {
                std::cout << "The relative residual norm is not implemented for the size " << nx <<"x"<< ny << "x"<<nz << std::endl;
                std::cout << "Please add the size run_get_Norm in the testing lib and run it to obtain the relative residual norm" << std::endl;
                std::cout << "then add the obtained value to the HPCG_functions::getSymGS_rrNrom function" << std::endl;
                std::cout << "and re-run the benchmark" << std::endl;
                assert(false);
            }

        }

        double getSymGS_rrNorm_zero_init(int nx, int ny, int nz){

            // The relative residual norm for 2x2x2 is 0.03859589326112207
            // The relative residual norm for 4x4x4 is 0.1627214749610502
            // The relative residual norm for 8x8x8 is 0.1878644061539017
            // The relative residual norm for 16x16x16 is 0.1868789912880421
            // The relative residual norm for 32x32x32 is 0.2386720453340627
            // The relative residual norm for 64x64x64 is 0.3411755350583427
            // The relative residual norm for 128x64x64 is 0.375064237277454
            // The relative residual norm for 128x128x64 is 0.4186180854459645
            // The relative residual norm for 128x128x128 is 0.47348419196090

            if(nx == 2 and ny == 2 and nz == 2){
                return 0.03859589326112207;
            } else if(nx == 4 and ny == 4 and nz == 4){
                return 0.1627214749610502;
            } else if(nx == 8 and ny == 8 and nz == 8){
                return 0.1878644061539017;
            } else if(nx == 16 and ny == 16 and nz == 16){
                return 0.1868789912880421;
            } else if(nx == 24 and ny == 24 and nz == 24){
                return 0.2098583236701958;
            } else if(nx == 32 and ny == 32 and nz == 32){
                return 0.2386720453340627;
            } else if(nx == 64 and ny == 64 and nz == 64){
                return 0.3411755350583427;
            } else if(nx == 128 and ny == 64 and nz == 64){
                return 0.375064237277454;
            } else if(nx == 128 and ny == 128 and nz == 64){
                return 0.4186180854459645;
            } else if(nx == 128 and ny == 128 and nz == 128){
                return 0.47348419196090;
            } else if (nx == 256 and ny == 128 and nz == 128){
                return 0.51171735634656;
            } else if (nx == 256 and ny == 256 and nz == 128){
                return 0.5586609181757166;
            } else {
                std::cout << "The relative residual norm is not implemented for the size " << nx <<"x"<< ny << "x"<<nz << std::endl;
                std::cout << "Please add the size run_get_Norm in the testing lib and run it to obtain the relative residual norm" << std::endl;
                std::cout << "then add the obtained value to the HPCG_functions::getSymGS_rrNrom function" << std::endl;
                std::cout << "and re-run the benchmark" << std::endl;
                std::cout << "Returning -1.0, this will cause an assertion error." << std::endl;
                return -1.0;
            }
        }
    protected:
        int max_CG_iterations = 800;
        double CG_tolerance = 1e-9;
        int max_SymGS_iterations = 1;
        double SymGS_tolerance = 1e-6;
};

#endif // HPCGLIB_HPP