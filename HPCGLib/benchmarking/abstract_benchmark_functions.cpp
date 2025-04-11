#include "benchmark.hpp"
#include <sstream>

// these function calls the abstract function the required number of times and records the time
// again we have method overloading for different matrix types

void bench_CG(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    )
{
    int num_iterations = implementation.getNumberOfIterations();
    
    // grab original data to store it
    std::vector<DataType> x_original(A.get_num_rows(), 0.0);
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    if(implementation.test_before_bench and not implementation.CG_file_based_tests_passed){
        // we do not have a baseline for CG
        // therefore we run the filebased tests
        bool test_passed = test_CG(implementation);
        implementation.CG_file_based_tests_passed = test_passed;

        // std::cout << "CG tested for implementation " << implementation.version_name << std::endl;

        if (not test_passed){
            num_iterations = 0;
        }
    }

    // we need a few more parameters for the CG function
    int n_iters = 0;
    double normr;
    double normr0;

    // we do both with and without preconditioning

    if( // the MG preconditioner can only be applied for matrices divisible by 8 and the result needs to be bigger than 2
        A.get_nx() % 8 == 0 and
        A.get_ny() % 8 == 0 and
        A.get_nz() % 8 == 0 and
        A.get_nx() / 8 > 2 and
        A.get_ny() / 8 > 2 and
        A.get_nz() / 8 > 2
    ){
        implementation.doPreconditioning = true;
        for(int i = 0; i < num_iterations; i++){
            // restore original x
            CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
            
            timer.startTimer();
            implementation.compute_CG(
                A,
                y_d, x_d,
                n_iters, normr, normr0
            );
            timer.stopTimer("compute_CG");
            std::cout << "CG took " << n_iters << " iterations for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz()<< " and implementation " << implementation.version_name << std::endl;

        }
        // add the CG iterations to the timer
        std::ostringstream oss;
        oss << "CG iterations: " << n_iters;
        std::string n_iters_string = oss.str();
        timer.add_additional_parameters(n_iters_string);
        CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    } else{
        std::cout << "Skipping CG Preconditioned bench for matrix with dimensions " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << " not divisible by 8 or too small for MG" << std::endl;
    }

    // now without preconditioning (can always be done)
    implementation.doPreconditioning = false;
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_CG(
            A,
            y_d, x_d,
            n_iters, normr, normr0
        );
        timer.stopTimer("compute_CG_noPreconditioning");
        // restore original x
        CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    }
}

void bench_CG(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    )
{
    int num_iterations = implementation.getNumberOfIterations();
    
    // grab original data to store it
    std::vector<DataType> x_original(A.get_num_rows(), 0.0);
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    if(implementation.test_before_bench and not implementation.CG_file_based_tests_passed){
        // we do not have a baseline for CG
        // therefore we run the filebased tests
        bool test_passed = test_CG(implementation);
        implementation.CG_file_based_tests_passed = test_passed;

        // std::cout << "CG tested for implementation " << implementation.version_name << std::endl;

        if (not test_passed){
            num_iterations = 0;
        }
    }

    // we need a few more parameters for the CG function
    int n_iters = 0;
    double normr;
    double normr0;

    // we do both with and without preconditioning

    if( // the MG preconditioner can only be applied for matrices divisible by 8 and the result needs to be bigger than 2
        A.get_nx() % 8 == 0 and
        A.get_ny() % 8 == 0 and
        A.get_nz() % 8 == 0 and
        A.get_nx() / 8 > 2 and
        A.get_ny() / 8 > 2 and
        A.get_nz() / 8 > 2
    ){
        implementation.doPreconditioning = true;
        for(int i = 0; i < num_iterations; i++){
            timer.startTimer();
            implementation.compute_CG(
                A,
                y_d, x_d,
                n_iters, normr, normr0
            );
            timer.stopTimer("compute_CG");
            std::cout << "CG took " << n_iters << " iterations for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz()<< " and implementation " << implementation.version_name << std::endl;
            // restore original x
            CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
        }
    } else{
        std::cout << "Skipping CG Preconditioned bench for matrix with dimensions " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << " not divisible by 8 or too small for MG" << std::endl;
    }

    // now without preconditioning (can always be done)
    implementation.doPreconditioning = false;
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_CG(
            A,
            y_d, x_d,
            n_iters, normr, normr0
        );
        timer.stopTimer("compute_CG_noPreconditioning");
        // restore original x
        CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    }
}


//this CG supports multi GPU
void bench_CG(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A,
    Halo * x_d, Halo * y_d, Problem *problem
    )
{
    int num_iterations = implementation.getNumberOfIterations();
    
    // grab original data to store it
    local_int_t size_halo = y_d->dimx * y_d->dimy * y_d->dimz;
    std::vector<DataType> x_original(size_halo, 0.0);
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d->x_d, size_halo * sizeof(DataType), cudaMemcpyDeviceToHost));

    // we need a few more parameters for the CG function
    int n_iters = 0;
    DataType normr;
    DataType normr0;

    // we do both with and without preconditioning

    if( // the MG preconditioner can only be applied for matrices divisible by 8 and the result needs to be bigger than 2
        A.get_nx() % 8 == 0 and
        A.get_ny() % 8 == 0 and
        A.get_nz() % 8 == 0 and
        A.get_nx() / 8 > 2 and
        A.get_ny() / 8 > 2 and
        A.get_nz() / 8 > 2
    ){
        implementation.doPreconditioning = true;
        for(int i = 0; i < num_iterations; i++){
            timer.startTimer();
            implementation.compute_CG(
                A,
                y_d, x_d,
                n_iters, normr, normr0,
                problem
            );
            timer.stopTimer("compute_CG");
            if(problem->rank == 0)std::cout << "CG took " << n_iters << " iterations for size " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz()<< " and implementation " << implementation.version_name << std::endl;
            // restore original x
            CHECK_CUDA(cudaMemcpy(x_d->x_d, x_original.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
        }
    } else{
        if(problem->rank == 0)std::cout << "Skipping CG Preconditioned bench for matrix with dimensions " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << " not divisible by 8 or too small for MG" << std::endl;
    }

    // now without preconditioning (can always be done)
    implementation.doPreconditioning = false;
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_CG(
            A,
            y_d, x_d,
            n_iters, normr, normr0, problem
        );
        timer.stopTimer("compute_CG_noPreconditioning");
        // restore original x
        CHECK_CUDA(cudaMemcpy(x_d->x_d, x_original.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
    }
}


//this MG supprt multi GPU
void bench_MG(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A,
    Halo * x_d, Halo * y_d,
    Problem *problem
    )
{
    int num_iterations = implementation.getNumberOfIterations();

    // we can only bench MG if the matrix dimensions are divisible by 8 and the result is bigger than 2

    if(
        A.get_nx() % 8 != 0 or
        A.get_ny() % 8 != 0 or
        A.get_nz() % 8 != 0 or
        A.get_nx() / 8 < 3 or
        A.get_ny() / 8 < 3 or
        A.get_nz() / 8 < 3
    ){
        if(problem->rank == 0)std::cout << "Skipping MG bench for matrix with dimensions " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << " not divisible by 8 or too small for MG" << std::endl;
        return;
    }

    // obtain the original x vector
    local_int_t size_halo = y_d->dimx * y_d->dimy * y_d->dimz;
    std::vector<DataType> x_original(size_halo, 0.0);
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d->x_d, size_halo * sizeof(DataType), cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_MG(
            A,
            y_d, x_d,
            problem
        );
        timer.stopTimer("compute_MG");
    }

    // restore the original x vector
    CHECK_CUDA(cudaMemcpy(x_d->x_d, x_original.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
}

void bench_MG(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    )
{
    int num_iterations = implementation.getNumberOfIterations();

    // we can only bench MG if the matrix dimensions are divisible by 8 and the result is bigger than 2

    if(
        A.get_nx() % 8 != 0 or
        A.get_ny() % 8 != 0 or
        A.get_nz() % 8 != 0 or
        A.get_nx() / 8 < 3 or
        A.get_ny() / 8 < 3 or
        A.get_nz() / 8 < 3
    ){
        std::cout << "Skipping MG bench for matrix with dimensions " << A.get_nx() << "x" << A.get_ny() << "x" << A.get_nz() << " not divisible by 8 or too small for MG" << std::endl;
        return;
    }

    // obtain the original x vector
    std::vector<DataType> x_original(A.get_num_rows(), 0.0);
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    if (implementation.test_before_bench and not implementation.MG_file_based_tests_passed){
        
        // there is no MG baseline so we test against the filebased tests
        bool test_passed = test_MG(implementation);
        implementation.MG_file_based_tests_passed = test_passed;

        if (not test_passed){
            num_iterations = 0;
        }
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_MG(
            A,
            y_d, x_d
        );
        timer.stopTimer("compute_MG");
    }

    // restore the original x vector
    CHECK_CUDA(cudaMemcpy(x_d, x_original.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
}

// this SPMV supports CSR matrixes
void bench_SPMV(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    )
{

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<DataType> y(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(y.data(), y_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
    // we always test against cusparse$
        cuSparse_Implementation<DataType> baseline;
        bool test_failed = !test_SPMV(
            baseline, implementation,
            A, x_d);
        if (test_failed){
            num_iterations = 0;
        }
    }


    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            x_d, y_d
        );
        timer.stopTimer("compute_SPMV");
    }

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
}

// this SPMV supports striped matrixes which requires CSR for metadata and testing
void bench_SPMV(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    ){

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<DataType> y(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(y.data(), y_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
    // we always test against cusparse
        cuSparse_Implementation<DataType> baseline;
    
        // test the SPMV function
        bool test_failed = !test_SPMV(
            baseline, implementation,
            A,
            x_d
            );

        if (test_failed)
        {
            num_iterations = 0;
        }
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            x_d, y_d
        );
        timer.stopTimer("compute_SPMV");
    }

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(y_d, y.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
}

// this SPMV supports multi GPU
void bench_SPMV(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A,
    Halo * x_d, Halo * y_d,
    Problem *problem
    ){

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    local_int_t size_halo = y_d->dimx * y_d->dimy * y_d->dimz;
    std::vector<DataType> y(size_halo, 0.0);

    CHECK_CUDA(cudaMemcpy(y.data(), y_d->x_d, size_halo * sizeof(DataType), cudaMemcpyDeviceToHost));

    int num_iterations = implementation.getNumberOfIterations();

    // testing not possible for multi GPU
    /*
    if (implementation.test_before_bench){
    // we always test against cusparse
        cuSparse_Implementation<DataType> baseline;
    
        // test the SPMV function
        bool test_failed = !test_SPMV(
            baseline, implementation,
            A,
            x_d
            );

        if (test_failed)
        {
            num_iterations = 0;
        }
    } */

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_SPMV(
            A,
            x_d, y_d, problem
        );
        timer.stopTimer("compute_SPMV");
    }

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(y_d->x_d, y.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
}

void bench_Dot(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType * result_d
    ){
    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
        // note that for the dot product the cuSparse implementation is an instanciation of warp reduction. ehem.
        cuSparse_Implementation<DataType> baseline;
        bool test_failed = !test_Dot(
            baseline, implementation,
            A,
            x_d, y_d
        );

        if (test_failed){
            num_iterations = 0;
        }            
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_Dot(
            A,
            x_d, y_d, result_d
        );
        timer.stopTimer("compute_Dot");
    }
    
}

// this Dot supports multi GPU
void bench_Dot(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    Halo * x_d, Halo * y_d, DataType * result_d,
    Problem *problem
    ){
    int num_iterations = implementation.getNumberOfIterations();

    //not possible for multi GPU
    /*
    if (implementation.test_before_bench){
        // note that for the dot product the cuSparse implementation is an instanciation of warp reduction. ehem.
        cuSparse_Implementation<DataType> baseline;
        bool test_failed = !test_Dot(
            baseline, implementation,
            A,
            x_d, y_d
        );

        if (test_failed){
            num_iterations = 0;
        }            
    }
    */

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_Dot(
            x_d, y_d, result_d
        );
        timer.stopTimer("compute_Dot");
    }
    
}

void bench_Dot(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType * result_d
    ){
    int num_iterations = implementation.getNumberOfIterations();

    if (implementation.test_before_bench){
        // note that for the dot product the cuSparse implementation is an instanciation of warp reduction. ehem.
        cuSparse_Implementation<DataType> baseline;
        bool test_failed = !test_Dot(
            implementation,
            A,
            x_d, y_d
        );

        if (test_failed){
            num_iterations = 0;
        }            
    }

    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_Dot(
            A,
            x_d, y_d, result_d
        );
        timer.stopTimer("compute_Dot");
    }
    
}

void bench_WAXPBY(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d, DataType * w_d,
    DataType alpha, DataType beta
    ){

    int num_iterations = implementation.getNumberOfIterations();
    
    // grab original value of w_d
    std::vector<DataType> w(A.get_num_rows(), 0.0);
    CHECK_CUDA(cudaMemcpy(w.data(), w_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    if(implementation.test_before_bench){

        bool test_failed = !test_WAXPBY(
            implementation, A, x_d, y_d, alpha, beta
        );

        // restore original value of w_d
        CHECK_CUDA(cudaMemcpy(w_d, w.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));

        if (test_failed){
            num_iterations = 0;
        }
    }
    
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_WAXPBY(
            A,
            x_d, y_d, w_d, alpha, beta
        );
        timer.stopTimer("compute_WAXPBY");

        // restore original value of w_d
        CHECK_CUDA(cudaMemcpy(w_d, w.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    }
}

// this WAXPBY supports multi GPU
void bench_WAXPBY(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    Halo * x_d, Halo * y_d, Halo * w_d,
    DataType alpha, DataType beta,
    Problem *problem
    ){

    int num_iterations = implementation.getNumberOfIterations();
    
    // grab original value of w_d
    local_int_t size_halo = w_d->dimx * w_d->dimy * w_d->dimz;
    std::vector<DataType> w(size_halo, 0.0);
    CHECK_CUDA(cudaMemcpy(w.data(), w_d->x_d, size_halo * sizeof(DataType), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.compute_WAXPBY(
            x_d, y_d, w_d, alpha, beta, problem, false
        );
        timer.stopTimer("compute_WAXPBY");

        // restore original value of w_d
        CHECK_CUDA(cudaMemcpy(w_d->x_d, w.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
    }
}


void bench_SymGS(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,   
    DataType * x_d, DataType * y_d
    )
{
    int num_iterations = implementation.getNumberOfIterations();

    // we need the following parameters to compute the norm
    int num_rows = A.get_num_rows();
    int num_cols = A.get_num_cols();

    // y_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<DataType> x(A.get_num_rows(), 0.0);

    double norm0 = L2_norm_for_SymGS(A, x_d, y_d);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));

    if (implementation.test_before_bench){
        cuSparse_Implementation<DataType> baseline;

        bool test_failed = !test_SymGS(
            baseline, implementation,
            A, x_d, y_d);
        if (test_failed){
            num_iterations = 0;
        }
    }

    // in case it is a norm based SymGS (we do more than one iteration)
    // we need to store and adjust the number of max iterations
    int original_max_iter = implementation.get_maxSymGSIters();
    implementation.set_maxSymGSIters(10);

    std::cout << "num iterations for implementation " << implementation.version_name << ": " << implementation.get_maxSymGSIters() << std::endl;

    for(int i = 0; i < num_iterations; i++){
        // always write the original x back into x_d
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_cols() * sizeof(DataType), cudaMemcpyHostToDevice));
        timer.startTimer();
        implementation.compute_SymGS(
            A,
            x_d, y_d
        );
        timer.stopTimer("compute_SymGS");
    }
    double normPostExe = L2_norm_for_SymGS(A, x_d, y_d);

    // greb da norm and store it in additional infos
    double norm = normPostExe / norm0;

    std::cout << "norm: " << norm << " for implementation " << implementation.version_name << std::endl;

    std::ostringstream oss;
    oss << "RR Norm: " << norm;
    std::string norm_string = oss.str();
    timer.add_additional_parameters(norm_string);

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    
    // store the original number of iterations
    std::cout << "original max iter: " << original_max_iter << std::endl;
    implementation.set_maxSymGSIters(original_max_iter);

}

void bench_SymGS(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A,
    DataType * x_d, DataType * y_d
    )
{   
    int num_iterations = implementation.getNumberOfIterations();

    // std::cout << "benching symgs for " << num_iterations << " iterations" << std::endl;

    // x_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    std::vector<DataType> x(A.get_num_rows(), 0.0);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d, A.get_num_rows() * sizeof(DataType), cudaMemcpyDeviceToHost));   

    if(implementation.test_before_bench){

        cuSparse_Implementation<DataType> baseline;           

        bool test_failed = !test_SymGS(
            baseline, implementation,
            A,
            y_d);

        if (test_failed){
            num_iterations = 0;
        }
    }

    double norm0 = implementation.L2_norm_for_SymGS(A, x_d, y_d);

    // for normbased implementations we need to make sure the maximum number of iterations performed by symGS is enough
    int original_max_symgs_iterations = implementation.get_maxSymGSIters();

    implementation.set_maxSymGSIters(10);

    std::cout << "num iterations for implementation " << implementation.version_name << ": " << implementation.get_maxSymGSIters() << std::endl;

    for (int i = 0; i < num_iterations; i++){
        // std::cout<< "Iteration: " << i << std::endl;
        // std::cout<< "Num iterations: " << num_iterations << std::endl;
        // copy original x into x_d
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_cols() * sizeof(DataType), cudaMemcpyHostToDevice));
        timer.startTimer();
        implementation.compute_SymGS( A, x_d, y_d);
        timer.stopTimer("compute_SymGS");
    }

    double normPostExe = implementation.L2_norm_for_SymGS(A, x_d, y_d);

    // greb da norm and store it in additional infos
    double norm = normPostExe / norm0;

    std::cout << "norm: " << norm << " for implementation " << implementation.version_name << std::endl;

    std::ostringstream oss;
    oss << "normi/norm0: " << norm;
    std::string norm_string = oss.str();
    timer.add_additional_parameters(norm_string);


    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(x_d, x.data(), A.get_num_rows() * sizeof(DataType), cudaMemcpyHostToDevice));
    // store the original number of iterations


    // restore the original number of iterations
    std::cout << "original max iter: " << original_max_symgs_iterations << std::endl;

    implementation.set_maxSymGSIters(original_max_symgs_iterations);
}

// this SymGS supports multi GPU
void bench_SymGS(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A,
    Halo * x_d, Halo * y_d,
    Problem *problem
    )
{   
    int num_iterations = implementation.getNumberOfIterations();

    // std::cout << "benching symgs for " << num_iterations << " iterations" << std::endl;

    // x_d is the output vector, hence we need to store the original and write the original back after the benchmarking
    local_int_t size_halo = y_d->dimx * y_d->dimy * y_d->dimz;
    std::vector<DataType> x(size_halo, 0.0);

    CHECK_CUDA(cudaMemcpy(x.data(), x_d->x_d, size_halo * sizeof(DataType), cudaMemcpyDeviceToHost));   

     // for normbased implementations we need to make sure the maximum number of iterations performed by symGS is enough
     int original_max_symgs_iterations = implementation.get_maxSymGSIters();
     if(implementation.norm_based){
         implementation.set_maxSymGSIters(10);
     }

    for (int i = 0; i < num_iterations; i++){
        // std::cout<< "Iteration: " << i << std::endl;
        // std::cout<< "Num iterations: " << num_iterations << std::endl;
        // copy original x into x_d
        CHECK_CUDA(cudaMemcpy(x_d->x_d, x.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
        timer.startTimer();
        implementation.compute_SymGS( A, x_d, y_d, problem);
        timer.stopTimer("compute_SymGS");
    }

    //TODO: not impleemnted
    /*
    // greb da norm and store it in additional infos
    double norm = relative_residual_norm_for_SymGS(
    A,
    x_d, y_d);

    std::ostringstream oss;
    oss << "RR Norm: " << norm;
    std::string norm_string = oss.str();
    timer.add_additional_parameters(norm_string);
*/

    // copy the original vector back
    CHECK_CUDA(cudaMemcpy(x_d->x_d, x.data(), size_halo * sizeof(DataType), cudaMemcpyHostToDevice));
    // store the original number of iterations
    if(implementation.norm_based){
        implementation.set_maxSymGSIters(original_max_symgs_iterations);
    }

    // restore the original number of iterations
    implementation.set_maxSymGSIters(original_max_symgs_iterations);
}


// function to benchmark the exchange halo function
void bench_ExchangeHalo(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    Halo *x_d,
    Problem *problem
    )
{   
    int num_iterations = implementation.getNumberOfIterations();

    for (int i = 0; i < num_iterations; i++){
        timer.startTimer();
        implementation.ExchangeHalo(x_d, problem);
        timer.stopTimer("ExchangeHalo");
    }

}


// this function allows us to run the whole abstract benchmark
// we have method overloading to support different matrix types

// this version supports CSR
void bench_Implementation(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    sparse_CSR_Matrix<DataType> & A,
    DataType * a_d, DataType * b_d, // a & b are random vectors
    DataType * x_d, DataType * y_d // x & y are vectors as used in HPCG
    )
{   
    // we want to make sure the vectors are not changed, so we grab the first 100 elements
    int num_sanity_elements = 100;
    std::vector<DataType> a_original(num_sanity_elements);
    std::vector<DataType> b_original(num_sanity_elements);
    std::vector<DataType> x_original(num_sanity_elements);
    std::vector<DataType> y_original(num_sanity_elements);

    // copy the first 100 elements of the vectors
    CHECK_CUDA(cudaMemcpy(a_original.data(), a_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(b_original.data(), b_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_original.data(), y_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));

    // make a vector of vectors
    std::vector<std::vector<DataType>> original_vectors = {a_original, b_original, x_original, y_original};
    std::vector<DataType*> vectors_d = {a_d, b_d, x_d, y_d};

    if(implementation.SPMV_implemented){
        bench_SPMV(implementation, timer, A, a_d, y_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }
    if(implementation.SymGS_implemented){
        bench_SymGS(implementation, timer, A, x_d, y_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }

    // bench_SPMV(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    // bench_SymGS(implementation, timer, A, A_row_ptr_d, A_col_idx_d, A_values_d, x_d, y_d);
    // other functions to be benchmarked
}


// this version supports striped matrixes
void bench_Implementation(
    HPCG_functions<DataType>& implementation,
    Timer& timer,
    striped_Matrix<DataType> & A, // we need to pass the CSR matrix for metadata and potential testing
    DataType * a_d, DataType * b_d, // a & b are random vectors
    DataType * x_d, DataType * y_d, // x & y are vectors as used in HPCG
    DataType * result_d ,  // result is used for the dot product (it is a scalar)
    DataType alpha, DataType beta
){
   
    // we want to make sure the vectors are not changed, so we grab the first 100 elements
    int num_sanity_elements = 100;
    std::vector<DataType> a_original(num_sanity_elements);
    std::vector<DataType> b_original(num_sanity_elements);
    std::vector<DataType> x_original(num_sanity_elements);
    std::vector<DataType> y_original(num_sanity_elements);

    // copy the first 100 elements of the vectors
    CHECK_CUDA(cudaMemcpy(a_original.data(), a_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(b_original.data(), b_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(x_original.data(), x_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y_original.data(), y_d, num_sanity_elements * sizeof(DataType), cudaMemcpyDeviceToHost));
    
    // make a vector of vectors
    std::vector<std::vector<DataType>> original_vectors = {a_original, b_original, x_original, y_original};
    std::vector<DataType*> vectors_d = {a_d, b_d, x_d, y_d};
    
    // we do one sanity check prior to the benchmarking (just for my sanity and to make debugging easier)
    sanity_check_vectors(vectors_d, original_vectors);

    std::cout << "Bench SPMV" << std::endl;
    if(implementation.SPMV_implemented){
        bench_SPMV(implementation, timer, A, a_d, y_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }

    std::cout << "Bench Dot" << std::endl;
    if(implementation.Dot_implemented){
        bench_Dot(implementation, timer, A, a_d, b_d, result_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }
    // std::cout << "Bench SymGS" << std::endl;
    // if(implementation.SymGS_implemented){
    //     bench_SymGS(implementation, timer, A, x_d, y_d);
    //     sanity_check_vectors(vectors_d, original_vectors);
    // }
    std::cout << "Bench WAXPBY" << std::endl;
    if(implementation.WAXPBY_implemented){
        bench_WAXPBY(implementation, timer, A, a_d, b_d, y_d, alpha, beta);
        sanity_check_vectors(vectors_d, original_vectors);
    }
    std::cout << "Bench CG" << std::endl;
    if(implementation.CG_implemented){
        bench_CG(implementation, timer, A, x_d, y_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }
    std::cout << "Bench MG" << std::endl;
    if(implementation.MG_implemented){
        bench_MG(implementation, timer, A, x_d, y_d);
        sanity_check_vectors(vectors_d, original_vectors);
    }
}

void bench_Implementation(
    striped_multi_GPU_Implementation<DataType>& implementation,
    Timer& timer,
    striped_partial_Matrix<DataType> & A, // we need to pass the CSR matrix for metadata and potential testing
    Halo * a_d, Halo * b_d, // a & b are random vectors
    Halo * x_d, Halo * y_d, // x & y are vectors as used in HPCG
    DataType * result_d,  // result is used for the dot product (it is a scalar)
    DataType alpha, DataType beta,
    Problem *problem,
    const std::string& benchFilter
){

    // Helper lambda: if benchFilter equals "ALL" or contains the given benchmark name, return true.
    auto runBenchmark = [&benchFilter](const std::string& name) -> bool {
        return (benchFilter == "ALL" || benchFilter.find(name) != std::string::npos);
    };

    if (runBenchmark("SPMV")) {
        if(problem->rank == 0) std::cout << "Bench SPMV" << std::endl;
        if(implementation.SPMV_implemented){
            bench_SPMV(implementation, timer, A, a_d, y_d, problem);
        }
    }

    if (runBenchmark("DOT")) {
        if(problem->rank == 0) std::cout << "Bench Dot" << std::endl;
        if(implementation.Dot_implemented){
            bench_Dot(implementation, timer, a_d, b_d, result_d, problem);
        }
    }

    if (runBenchmark("SYMGS")) {
        if(problem->rank == 0) std::cout << "Bench SymGS" << std::endl;
        if(implementation.SymGS_implemented){
            bench_SymGS(implementation, timer, A, x_d, y_d, problem);
        }
    }

    if (runBenchmark("WAXPBY")) {
        if(problem->rank == 0) std::cout << "Bench WAXPBY" << std::endl;
        if(implementation.WAXPBY_implemented){
            bench_WAXPBY(implementation, timer, a_d, b_d, y_d, alpha, beta, problem);
        }
    }

    if (runBenchmark("CG")) {
        if(problem->rank == 0) std::cout << "Bench CG" << std::endl;
        if(implementation.CG_implemented){
            bench_CG(implementation, timer, A, x_d, y_d, problem);
        }
    }

    if (runBenchmark("MG")) {
        if(problem->rank == 0) std::cout << "Bench MG" << std::endl;
        if(implementation.MG_implemented){
            bench_MG(implementation, timer, A, x_d, y_d, problem);
        }
    }

    if (runBenchmark("HALO")) {
        if(problem->rank == 0) std::cout << "Bench Halo Exchange" << std::endl;
        bench_ExchangeHalo(implementation, timer, x_d, problem);
    }

}
