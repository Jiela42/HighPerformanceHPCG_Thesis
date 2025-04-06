#!/bin/bash

spack load hip@6.2.1
spack load cmake@3.30.5
spack load hipsparse@6.2.1
spack load hipblas@6.2.1
spack load rocm-cmake@6.2.1
spack load rocm-core@6.2.1
spack load rocsparse@6.2.1
spack load rocblas@6.2.1
spack load rocthrust@6.2.1
spack load rccl@6.2.1
spack load mpich@4.2.3/dfsmhyo
spack load rocprim@6.2.1


export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_CXX=hipcc



#export MPI_INCLUDE=/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/mpich-4.2.3-dfsmhyouvtg5jag7pzsrxzh4d7jlfhmi/include
#export MPI_LIB=/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/mpich-4.2.3-dfsmhyouvtg5jag7pzsrxzh4d7jlfhmi/lib
#export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH
