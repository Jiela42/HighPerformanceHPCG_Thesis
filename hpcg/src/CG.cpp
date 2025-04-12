
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file CG.cpp

 HPCG routine
 */

#include <fstream>

#include <cmath>

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"

#include "HPCG_versions/striped_multi_GPU.cuh"


// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning) {
  //doPreconditioning = false;
  //zero the vector
  SetHaloZeroGPU(A.x_halo);
  cudaDeviceSynchronize();
      
  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;


  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  local_int_t nrow = A.localNumberOfRows;
  // Vector & r = data.r; // Residual vector
  // Vector & z = data.z; // Preconditioned residual vector
  // Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  // Vector & Ap = data.Ap;

  Halo * r_halo = data.r_halo;
  Halo * z_halo = data.z_halo;
  Halo * p_halo = data.p_halo;
  Halo * Ap_halo = data.Ap_halo;
  DataType * normr_d = data.normr_d;
  DataType * pAp_d = data.pAp_d;
  DataType * rtz_d = data.rtz_d;

  striped_multi_GPU_Implementation<DataType> * implementation = A.implementation;
  Problem * problem = A.problem;
  striped_partial_Matrix<DataType> * A_striped = A.A_striped;
  Halo * b_halo = A.b_halo;
  Halo * x_halo = A.x_halo;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  //CopyVector(x, p);
  cudaMemcpy(p_halo->x_d, x_halo->x_d, x_halo->dimx * x_halo->dimy * x_halo->dimz * sizeof(DataType), cudaMemcpyDeviceToDevice);
  //TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
  TICK(); 
  implementation->compute_SPMV(*A_striped, p_halo, Ap_halo, problem);
  implementation->ExchangeHalo(Ap_halo, problem);
  TOCK(t3);
  //TICK(); ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);  TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); 
  implementation->compute_WAXPBY(b_halo, Ap_halo, r_halo, 1.0, -1.0, problem, false);
  implementation->ExchangeHalo(r_halo, problem);
  TOCK(t2);
  //TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
  TICK(); 
  implementation->compute_Dot(r_halo, r_halo, normr_d);
  cudaMemcpy(&normr, normr_d, sizeof(DataType), cudaMemcpyDeviceToHost);
  TOCK(t1);
  normr = sqrt(normr);
  #ifdef HPCG_DEBUG
  if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
  #endif
  
  // Record initial residual for convergence testing
  normr0 = normr;
  
  // Start iterations
  // Convergence check accepts an error of no more than 6 significant digits of tolerance
  for (int k=1; k<=max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++ ) {
    TICK();
    if (doPreconditioning){
      //ComputeMG(A, r, z); // Apply preconditioner
      implementation->compute_MG(*A_striped, r_halo, z_halo, problem); // Apply preconditioner
      implementation->ExchangeHalo(z_halo, problem);
    }else{
      //CopyVector (r, z); // copy r to z (no preconditioning)
      implementation->compute_WAXPBY(r_halo, r_halo, z_halo, 1.0, 0.0, problem, false);
      implementation->ExchangeHalo(z_halo, problem);
    }
    TOCK(t5); // Preconditioner apply time

    if (k == 1) {
      //TICK(); ComputeWAXPBY(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized); TOCK(t2); // Copy Mr to p
      TICK(); 
        implementation->compute_WAXPBY(z_halo, z_halo, p_halo, 1.0, 0.0, problem, false);
        implementation->ExchangeHalo(p_halo, problem);
      TOCK(t2);
      //TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
      TICK(); 
        implementation->compute_Dot(r_halo, z_halo, rtz_d);
        cudaMemcpy(&rtz, rtz_d, sizeof(DataType), cudaMemcpyDeviceToHost);
      TOCK(t1); // rtz = r'*z
    } else {
      oldrtz = rtz;
      //TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
      TICK(); 
        implementation->compute_Dot(r_halo, z_halo, rtz_d);
        cudaMemcpy(&rtz, rtz_d, sizeof(DataType), cudaMemcpyDeviceToHost);
      TOCK(t1); // rtz = r'*z
      beta = rtz/oldrtz;
      //TICK(); ComputeWAXPBY (nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);  TOCK(t2); // p = beta*p + z
      TICK(); 
        implementation->compute_WAXPBY(z_halo, p_halo, p_halo, 1.0, beta, problem, false);
        implementation->ExchangeHalo(p_halo, problem);
      TOCK(t2); // p = beta*p + z
    }

    //TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); 
        implementation->compute_SPMV(*A_striped, p_halo, Ap_halo, problem); 
        implementation->ExchangeHalo(Ap_halo, problem);
    TOCK(t3); // Ap = A*p
    //TICK(); ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized); TOCK(t1); // alpha = p'*Ap
    TICK(); 
        implementation->compute_Dot(p_halo, Ap_halo, pAp_d);
        cudaMemcpy(&pAp, pAp_d, sizeof(DataType), cudaMemcpyDeviceToHost);
    TOCK(t1); // alpha = p'*Ap
    alpha = rtz/pAp;
    //TICK(); ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized);// x = x + alpha*p
    //        ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);  TOCK(t2);// r = r - alpha*Ap
    TICK(); 
    implementation->compute_WAXPBY(x_halo, p_halo, x_halo, 1.0, alpha, problem, false);
    implementation->ExchangeHalo(x_halo, problem);
    implementation->compute_WAXPBY(r_halo, Ap_halo, r_halo, 1.0, -alpha, problem, false);
    implementation->ExchangeHalo(r_halo, problem);
    TOCK(t2);// r = r - alpha*Ap
    //TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
    TICK(); 
        implementation->compute_Dot(r_halo, r_halo, normr_d);
        cudaMemcpy(&normr, normr_d, sizeof(DataType), cudaMemcpyDeviceToHost);
    TOCK(t1);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;
  }

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
//#ifndef HPCG_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  // if(problem->rank == 0){
  //   printf("Opt took %d iterations and reach norm normr/normr0 = %e doPreconditioning=%d\n", niters, normr/normr0, doPreconditioning);
  // }
  return 0;
  
}
