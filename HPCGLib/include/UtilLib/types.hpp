// types.hpp
#ifndef TYPES_HPP
#define TYPES_HPP

typedef long local_int_t;
typedef long global_int_t;

using DataType = double;

#define MPIDataType MPI_DOUBLE
#define NUMBER_NEIGHBORS 26
#define STREAMS_PER_HALO 29 // stream 0-25: data extraction/injection, stream 26: border computation, stream 27: interior computation, stream 28: COO extraction

#define CHECK_MPI(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
      printf("Failed: MPI error %s:%d '%d'\n",        \
          __FILE__,__LINE__, e);   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

#define CHECK_NCCL(cmd) do {                         \
ncclResult_t r = cmd;                             \
if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

#endif // TYPES_HPP