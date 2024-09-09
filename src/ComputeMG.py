'''
Reference Implementation of the ComputeMG function in HPCG


  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG

int ComputeMG_ref(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}

'''

import torch

def computeMG(A: torch.sparse.Tensor, r: torch.Tensor, x: torch.Tensor, depth: int) -> int:

    # so there is a lot of stuff going on with mgData in the original. But it's already preprocessed there
    # so we can just assume that it's already done here and passed as this random thing called AmgData

    # in contrast to the reference implementation we compute the coarsening inside this function

    x = torch.zeros_like(x)
    ierr = 0

    if AmgData is not None:
        numberOfPresmootherSteps = AmgData.numberOfPresmootherSteps
        for i in range(numberOfPresmootherSteps):
            ierr += computeSYMGS(A, r, x)
        if ierr != 0:
            return ierr
        ierr = computeSPMV(A, x, AmgData.Axf)
        if ierr != 0:
            return ierr

        # Perform restriction operation using simple injection
        ierr = computeRestriction(A, r)
        if ierr != 0:
            return ierr
        
        # Dude! This thing essentially does not exist!!!
        ierr = computeMG(A.Ac, AmgData.rc, AmgData.xc)
        if ierr != 0:
            return ierr
        ierr = computeProlongation(A, x)
        if ierr != 0:
            return ierr

        numberOfPostsmootherSteps = AmgData.numberOfPostsmootherSteps
        for i in range(numberOfPostsmootherSteps):
            ierr += computeSYMGS(A, r, x)
        if ierr != 0:
            return ierr





    print("ComputeMG")
    return 0