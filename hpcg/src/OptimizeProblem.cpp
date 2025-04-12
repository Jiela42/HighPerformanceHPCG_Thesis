
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

#include "HPCG_versions/non_blocking_mpi_halo_exchange.cuh"
#include "UtilLib/hpcg_multi_GPU_utils.cuh"
#include "HPCG_versions/striped_multi_GPU.cuh"

int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  //retrieve data about setup
  Geometry * geom = A.geom;
  local_int_t nx = geom->nx;
  local_int_t ny = geom->ny;
  local_int_t nz = geom->nz;
  local_int_t npx = geom->npx;
  local_int_t npy = geom->npy;
  local_int_t npz = geom->npz;

  //setup the implementation
  non_blocking_mpi_Implementation<DataType> *MGPU_Implementation = new non_blocking_mpi_Implementation<DataType>();
  int argc; //only dummy, never accessed
  char *argv[1]; //only dummy, never accessed
  Problem *problem = MGPU_Implementation->init_comm(argc, argv, npx, npy, npz, nx, ny, nz);

  //print nx, ny, nz stroed in problem

  //initialize the matrix
  striped_partial_Matrix<DataType> *A_striped = new striped_partial_Matrix<DataType>(problem);

  //initialize the vectors
  Halo *b_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(b_halo, problem);
  DataType *b_d;
  CHECK_CUDA(cudaMalloc(&b_d, b.localLength * sizeof(DataType)));
  CHECK_CUDA(cudaMemcpy(b_d, b.values, b.localLength * sizeof(DataType), cudaMemcpyHostToDevice));
  InjectDataToHalo(b_halo, b_d);

  Halo *x_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(x_halo, problem);

  //initialize the MG data
  striped_partial_Matrix<DataType>* current_matrix = A_striped;
  for(int i = 0; i < 3; i++){
      current_matrix->initialize_coarse_matrix();
      current_matrix = current_matrix->get_coarse_Matrix();
  }

  //store our data into A
  A.implementation = MGPU_Implementation;
  A.problem = problem;
  A.A_striped = A_striped;
  A.b_halo = b_halo;
  A.x_halo = x_halo;

  //perepare data buffers
  Halo *r_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(r_halo, problem);
  data.r_halo = r_halo;
  Halo *z_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(z_halo, problem);
  data.z_halo = z_halo;
  Halo *p_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(p_halo, problem);
  data.p_halo = p_halo;
  Halo *Ap_halo = (Halo *) malloc(sizeof(Halo));
  InitHalo(Ap_halo, problem);
  //print halo information
  data.Ap_halo = Ap_halo;
  cudaMalloc(&data.normr_d, sizeof(DataType));
  cudaMalloc(&data.pAp_d, sizeof(DataType));
  cudaMalloc(&data.rtz_d, sizeof(DataType));

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
